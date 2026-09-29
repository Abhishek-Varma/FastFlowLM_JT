#include "models/qwen35/rai/aie_next/qwen35_rai_gguf.hpp"

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <string>

namespace flm::qwen35 {
namespace {

constexpr int kBlockWeights = 32;
constexpr int kBlockBytes = 34;
constexpr float kSteps = 15.0f;

std::uint64_t NeedUnsigned(const rai::GgufFile& file, std::string_view key) {
    if (!file.HasMetadata(key))
        throw std::runtime_error(std::string("the GGUF has no ") + std::string(key));
    return file.Unsigned(key);
}

double NeedNumber(const rai::GgufFile& file, std::string_view key) {
    if (!file.HasMetadata(key))
        throw std::runtime_error(std::string("the GGUF has no ") + std::string(key));
    return file.Number(key);
}

std::vector<std::int64_t> Expand(const std::vector<std::int64_t>& order, int width) {
    std::vector<std::int64_t> out(order.size() * static_cast<std::size_t>(width));
    for (std::size_t head = 0; head < order.size(); ++head) {
        for (int i = 0; i < width; ++i)
            out[head * static_cast<std::size_t>(width) + static_cast<std::size_t>(i)] =
                order[head] * width + i;
    }
    return out;
}

float HalfToFloat(std::uint16_t half) {
    const std::uint32_t sign = static_cast<std::uint32_t>(half & 0x8000u) << 16;
    const std::uint32_t exponent = (half >> 10) & 0x1fu;
    std::uint32_t fraction = half & 0x03ffu;
    std::uint32_t bits = 0;
    if (exponent == 0) {
        if (fraction == 0) {
            bits = sign;
        } else {
            int shift = 0;
            while ((fraction & 0x0400u) == 0) {
                fraction <<= 1;
                ++shift;
            }
            fraction &= 0x03ffu;
            bits = sign | (static_cast<std::uint32_t>(127 - 14 - shift) << 23) | (fraction << 13);
        }
    } else if (exponent == 0x1fu) {
        bits = sign | 0x7f800000u | (fraction << 13);
    } else {
        bits = sign | ((exponent + (127 - 15)) << 23) | (fraction << 13);
    }
    return std::bit_cast<float>(bits);
}

float Bf16Up(float value) {
    const auto bits = std::bit_cast<std::uint32_t>(value);
    auto truncated = static_cast<std::uint16_t>(bits >> 16);
    if ((bits & 0xFFFFu) != 0) truncated = static_cast<std::uint16_t>(truncated + 1);
    return std::bit_cast<float>(static_cast<std::uint32_t>(truncated) << 16);
}

std::uint16_t FloatToHalf(float value) {
    const auto bits = std::bit_cast<std::uint32_t>(value);
    const std::uint32_t sign = (bits >> 16) & 0x8000u;
    auto exponent = static_cast<std::int32_t>((bits >> 23) & 0xffu) - 127 + 15;
    std::uint32_t mantissa = bits & 0x7fffffu;
    if (exponent <= 0) {
        if (exponent < -10) return static_cast<std::uint16_t>(sign);
        mantissa |= 0x800000u;
        const auto shift = static_cast<std::uint32_t>(14 - exponent);
        std::uint32_t half = mantissa >> shift;
        const auto remainder = mantissa & ((1u << shift) - 1u);
        if (remainder > (1u << (shift - 1)) ||
            (remainder == (1u << (shift - 1)) && (half & 1u)))
            ++half;
        return static_cast<std::uint16_t>(sign | half);
    }
    if (exponent >= 31) return static_cast<std::uint16_t>(sign | 0x7c00u);
    std::uint32_t half = mantissa >> 13;
    const auto remainder = mantissa & 0x1fffu;
    if (remainder > 0x1000u || (remainder == 0x1000u && (half & 1u))) {
        ++half;
        if (half == 0x400u) {
            half = 0;
            ++exponent;
            if (exponent >= 31) return static_cast<std::uint16_t>(sign | 0x7c00u);
        }
    }
    return static_cast<std::uint16_t>(sign | (static_cast<std::uint32_t>(exponent) << 10) | half);
}

std::vector<float> Dequantize(std::span<const std::byte> blocks, int rows, int cols) {
    const int blocks_per_row = cols / kBlockWeights;
    if (static_cast<int>(blocks.size()) != rows * blocks_per_row * kBlockBytes)
        throw std::runtime_error("Q8_0 byte length does not match the matrix being refit");
    std::vector<float> out(static_cast<std::size_t>(rows) * static_cast<std::size_t>(cols));
    for (int row = 0; row < rows; ++row) {
        for (int block = 0; block < blocks_per_row; ++block) {
            const auto* encoded =
                blocks.data() + (static_cast<std::size_t>(row) * blocks_per_row + block) * kBlockBytes;
            std::uint16_t scale_bits = 0;
            std::memcpy(&scale_bits, encoded, sizeof(scale_bits));
            const float scale = HalfToFloat(scale_bits);
            for (int element = 0; element < kBlockWeights; ++element) {
                const auto code = static_cast<std::int8_t>(
                    std::to_integer<std::uint8_t>(encoded[2 + element]));
                out[static_cast<std::size_t>(row) * cols + block * kBlockWeights + element] =
                    scale * static_cast<float>(code);
            }
        }
    }
    return out;
}

}  // namespace

Qwen35Gguf::Qwen35Gguf(std::shared_ptr<rai::GgufFile> file, std::filesystem::path path,
                       Qwen35Config config, std::string lm_head, rai::GgufTensor embedding,
                       std::vector<std::int64_t> order)
    : file_(std::move(file)), path_(std::move(path)), config_(config), lm_head_(std::move(lm_head)),
      embedding_(std::move(embedding)), order_(std::move(order)) {}

std::shared_ptr<Qwen35Gguf> Qwen35Gguf::Open(const std::filesystem::path& path) {
    auto file = rai::GgufFile::Open(path, [](std::string_view key) {
        return key == "general.architecture" || key.starts_with("qwen35.");
    });
    const auto arch = file->String("general.architecture");
    if (arch != "qwen35")
        throw std::runtime_error("general.architecture is " + arch +
                                 "; this engine reads the dense Qwen3.5 conversion");

    Qwen35Config config;
    config.layers = static_cast<int>(NeedUnsigned(*file, "qwen35.block_count"));
    config.hidden = static_cast<int>(NeedUnsigned(*file, "qwen35.embedding_length"));
    config.intermediate = static_cast<int>(NeedUnsigned(*file, "qwen35.feed_forward_length"));
    config.q_heads = static_cast<int>(NeedUnsigned(*file, "qwen35.attention.head_count"));
    config.kv_heads = static_cast<int>(NeedUnsigned(*file, "qwen35.attention.head_count_kv"));
    config.head_dim = static_cast<int>(NeedUnsigned(*file, "qwen35.attention.key_length"));
    const auto value_dim = static_cast<int>(NeedUnsigned(*file, "qwen35.attention.value_length"));
    if (value_dim != config.head_dim)
        throw std::runtime_error("key_length and value_length differ");
    config.rope_dim = static_cast<int>(NeedUnsigned(*file, "qwen35.rope.dimension_count"));
    config.rope_theta = NeedNumber(*file, "qwen35.rope.freq_base");
    config.eps = NeedNumber(*file, "qwen35.attention.layer_norm_rms_epsilon");
    config.full_every = static_cast<int>(NeedUnsigned(*file, "qwen35.full_attention_interval"));
    config.conv_width = static_cast<int>(NeedUnsigned(*file, "qwen35.ssm.conv_kernel"));
    config.lin_k_dim = static_cast<int>(NeedUnsigned(*file, "qwen35.ssm.state_size"));
    config.lin_k_heads = static_cast<int>(NeedUnsigned(*file, "qwen35.ssm.group_count"));
    config.lin_v_heads = static_cast<int>(NeedUnsigned(*file, "qwen35.ssm.time_step_rank"));
    const auto inner = static_cast<int>(NeedUnsigned(*file, "qwen35.ssm.inner_size"));
    if (config.full_every <= 0 || config.lin_v_heads <= 0 || config.lin_k_heads <= 0 ||
        config.lin_v_heads % config.lin_k_heads != 0 || inner % config.lin_v_heads != 0)
        throw std::runtime_error("Qwen3.5 linear-attention metadata does not divide evenly");
    config.lin_v_dim = inner / config.lin_v_heads;

    const auto table = file->Tensor("token_embd.weight");
    if (table.ggml_type != rai::kGgmlTypeQ8_0 || table.shape.size() != 2 ||
        table.shape[1] != config.hidden)
        throw std::runtime_error("token_embd.weight is not Q8_0 [vocab, hidden]");
    config.vocab = static_cast<int>(table.shape[0]);
    config.tied_lm_head = !file->HasTensor("output.weight");
    const std::string lm_head = config.tied_lm_head ? "token_embd" : "output";
    if (file->Tensor(lm_head + ".weight").ggml_type != rai::kGgmlTypeQ8_0 ||
        file->Tensor("blk.0.attn_qkv.weight").ggml_type != rai::kGgmlTypeQ8_0)
        throw std::runtime_error("this engine packs Q8_0, which corelib refits to group 64");

    const int per = config.lin_v_heads / config.lin_k_heads;
    std::vector<std::int64_t> order(static_cast<std::size_t>(config.lin_v_heads));
    for (int head = 0; head < config.lin_v_heads; ++head)
        order[static_cast<std::size_t>(head)] = (head % per) * config.lin_k_heads + head / per;

    return std::shared_ptr<Qwen35Gguf>(new Qwen35Gguf(
        std::move(file), path, config, lm_head, table, std::move(order)));
}

std::span<const std::byte> Qwen35Gguf::Blocks(std::string_view base, std::int64_t k,
                                              std::int64_t n) const {
    const std::int64_t shape[] = {n, k};
    const auto tensor = file_->RequireQ8(std::string(base) + ".weight", shape);
    return tensor.bytes;
}

std::span<const float> Qwen35Gguf::F32(std::string_view name,
                                       std::span<const std::int64_t> shape) const {
    return file_->RequireF32(name, shape).values;
}

std::vector<std::byte> Qwen35Gguf::TakeRows(std::span<const std::byte> raw, std::int64_t n,
                                           std::int64_t k,
                                           std::span<const std::int64_t> rows) const {
    if (k % kBlockWeights != 0) throw std::runtime_error("Q8_0 width is not a whole number of blocks");
    const auto row_bytes = static_cast<std::size_t>(k / kBlockWeights * kBlockBytes);
    if (raw.size() != static_cast<std::size_t>(n) * row_bytes)
        throw std::runtime_error("Q8_0 row table is the wrong length");
    std::vector<std::byte> out(rows.size() * row_bytes);
    for (std::size_t i = 0; i < rows.size(); ++i) {
        const auto row = rows[i];
        if (row < 0 || row >= n) throw std::runtime_error("Q8_0 row reorder is out of range");
        std::memcpy(out.data() + i * row_bytes, raw.data() + static_cast<std::size_t>(row) * row_bytes,
                    row_bytes);
    }
    return out;
}

std::vector<std::int64_t> Qwen35Gguf::ChannelOrder() const {
    const auto qk = config_.lin_qk_dim() * 2;
    auto order = Expand(order_, config_.lin_v_dim);
    std::vector<std::int64_t> rows(static_cast<std::size_t>(config_.conv_channels()));
    for (int i = 0; i < qk; ++i) rows[static_cast<std::size_t>(i)] = i;
    for (std::size_t i = 0; i < order.size(); ++i)
        rows[static_cast<std::size_t>(qk) + i] = qk + order[i];
    return rows;
}

std::vector<std::byte> Qwen35Gguf::QkvBlocks(int layer) const {
    const auto base = "blk." + std::to_string(layer) + ".attn_qkv";
    const auto raw = Blocks(base, config_.hidden, config_.conv_channels());
    const auto rows = ChannelOrder();
    return TakeRows(raw, config_.conv_channels(), config_.hidden, rows);
}

std::vector<std::byte> Qwen35Gguf::ZBlocks(int layer) const {
    const auto raw = Blocks("blk." + std::to_string(layer) + ".attn_gate", config_.hidden,
                            config_.lin_value_dim());
    const auto rows = Expand(order_, config_.lin_v_dim);
    return TakeRows(raw, config_.lin_value_dim(), config_.hidden, rows);
}

std::vector<std::byte> Qwen35Gguf::GateBlocks(int layer, std::string_view which) const {
    const auto raw = Blocks("blk." + std::to_string(layer) + ".ssm_" + std::string(which),
                            config_.hidden, config_.lin_v_heads);
    return TakeRows(raw, config_.lin_v_heads, config_.hidden, order_);
}

std::vector<std::byte> Qwen35Gguf::OutBlocks(int layer) const {
    const auto raw = Blocks("blk." + std::to_string(layer) + ".ssm_out", config_.lin_value_dim(),
                            config_.hidden);
    const auto per = config_.lin_v_dim / kBlockWeights;
    const auto head_bytes = static_cast<std::size_t>(per * kBlockBytes);
    const auto row_bytes = head_bytes * static_cast<std::size_t>(config_.lin_v_heads);
    if (raw.size() != static_cast<std::size_t>(config_.hidden) * row_bytes)
        throw std::runtime_error("ssm_out Q8_0 length does not match its heads");
    std::vector<std::byte> out(raw.size());
    for (int row = 0; row < config_.hidden; ++row) {
        const auto* src = raw.data() + static_cast<std::size_t>(row) * row_bytes;
        auto* dst = out.data() + static_cast<std::size_t>(row) * row_bytes;
        for (int head = 0; head < config_.lin_v_heads; ++head) {
            std::memcpy(dst + static_cast<std::size_t>(head) * head_bytes,
                        src + static_cast<std::size_t>(order_[static_cast<std::size_t>(head)]) * head_bytes,
                        head_bytes);
        }
    }
    return out;
}

std::vector<float> Qwen35Gguf::PerHead(int layer, std::string_view name) const {
    const std::int64_t shape[] = {config_.lin_v_heads};
    const auto values = F32("blk." + std::to_string(layer) + "." + std::string(name), shape);
    std::vector<float> out(order_.size());
    for (std::size_t head = 0; head < order_.size(); ++head)
        out[head] = values[static_cast<std::size_t>(order_[head])];
    return out;
}

std::vector<float> Qwen35Gguf::ConvTaps(int layer) const {
    const std::int64_t shape[] = {config_.conv_channels(), config_.conv_width};
    const auto values = F32("blk." + std::to_string(layer) + ".ssm_conv1d.weight", shape);
    const auto rows = ChannelOrder();
    std::vector<float> out(values.size());
    for (std::size_t row = 0; row < rows.size(); ++row) {
        const auto src = static_cast<std::size_t>(rows[row]) * static_cast<std::size_t>(config_.conv_width);
        const auto dst = row * static_cast<std::size_t>(config_.conv_width);
        std::copy_n(values.begin() + static_cast<std::ptrdiff_t>(src), config_.conv_width,
                    out.begin() + static_cast<std::ptrdiff_t>(dst));
    }
    return out;
}

OnnxQ8 Q8ToOnnx(std::span<const std::byte> blocks, int k, int n, int group) {
    if (k % group != 0) throw std::runtime_error("K is not a whole number of quantization groups");
    auto weights = Dequantize(blocks, n, k);
    std::vector<float> transposed(static_cast<std::size_t>(k) * static_cast<std::size_t>(n));
    for (int row = 0; row < n; ++row) {
        for (int col = 0; col < k; ++col)
            transposed[static_cast<std::size_t>(col) * n + row] =
                weights[static_cast<std::size_t>(row) * k + col];
    }

    const int groups = k / group;
    std::vector<float> scales(static_cast<std::size_t>(groups) * static_cast<std::size_t>(n));
    std::vector<std::uint8_t> zeros(scales.size());
    std::vector<std::uint8_t> values(static_cast<std::size_t>(k) * static_cast<std::size_t>(n));
    for (int group_index = 0; group_index < groups; ++group_index) {
        for (int col = 0; col < n; ++col) {
            float lo = 0.0f;
            float hi = 0.0f;
            for (int row = 0; row < group; ++row) {
                const float value =
                    transposed[(static_cast<std::size_t>(group_index) * group + row) * n + col];
                lo = std::min(lo, value);
                hi = std::max(hi, value);
            }
            float scale = Bf16Up((hi - lo) / kSteps);
            std::uint8_t zero = 8;
            if (scale == 0.0f) {
                scale = 1.0f;
            } else {
                zero = static_cast<std::uint8_t>(
                    std::clamp(std::floor(-lo / scale + 0.5f), 0.0f, kSteps));
            }
            scales[static_cast<std::size_t>(group_index) * n + col] = scale;
            zeros[static_cast<std::size_t>(group_index) * n + col] = zero;
            for (int row = 0; row < group; ++row) {
                const auto at = (static_cast<std::size_t>(group_index) * group + row) * n + col;
                const float code = std::floor(transposed[at] / scale + 0.5f) + static_cast<float>(zero);
                values[at] = static_cast<std::uint8_t>(std::clamp(code, 0.0f, kSteps));
            }
        }
    }

    OnnxQ8 onnx;
    onnx.qweight.resize(static_cast<std::size_t>(n) * static_cast<std::size_t>(k / 2));
    onnx.scales.resize(static_cast<std::size_t>(n) * static_cast<std::size_t>(groups));
    onnx.qzeros.resize(static_cast<std::size_t>(n) * static_cast<std::size_t>(groups / 2));
    for (int row = 0; row < n; ++row) {
        for (int col = 0; col < k; col += 2) {
            const auto low = values[static_cast<std::size_t>(col) * n + row];
            const auto high = values[static_cast<std::size_t>(col + 1) * n + row];
            onnx.qweight[static_cast<std::size_t>(row) * (k / 2) + col / 2] =
                static_cast<std::uint8_t>(low | (high << 4));
        }
        for (int group_index = 0; group_index < groups; ++group_index) {
            onnx.scales[static_cast<std::size_t>(row) * groups + group_index] =
                FloatToHalf(scales[static_cast<std::size_t>(group_index) * n + row]);
        }
        for (int group_index = 0; group_index < groups; group_index += 2) {
            const auto low = zeros[static_cast<std::size_t>(group_index) * n + row];
            const auto high = zeros[static_cast<std::size_t>(group_index + 1) * n + row];
            onnx.qzeros[static_cast<std::size_t>(row) * (groups / 2) + group_index / 2] =
                static_cast<std::uint8_t>(low | (high << 4));
        }
    }
    return onnx;
}

}  // namespace flm::qwen35
