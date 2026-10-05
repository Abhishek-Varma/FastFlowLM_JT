/// \file qwen35_rai_gguf.hpp
/// \brief A Qwen3.5 Q8_0 GGUF, with the linear-attention V heads back in HF order
#pragma once

#include "rai/gguf_file.hpp"

#include <array>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace flm::qwen35 {

inline constexpr int kPrefillPdi = 9;
inline constexpr int kTokenPdi = 19;
inline constexpr int kGroup = 64;
inline constexpr int kMaxSequenceLength = 4096;
inline constexpr int kSoftplusOperand = 64;
inline constexpr int kRequantizeThreads = 1;
inline constexpr std::array<int, 2> kEosIds{248046, 248044};

struct Qwen35Config {
    int layers = 0;
    int hidden = 0;
    int intermediate = 0;
    int q_heads = 0;
    int kv_heads = 0;
    int head_dim = 0;
    int rope_dim = 0;
    double rope_theta = 0;
    double eps = 0;
    int full_every = 0;
    int conv_width = 0;
    int lin_k_heads = 0;
    int lin_v_heads = 0;
    int lin_k_dim = 0;
    int lin_v_dim = 0;
    int vocab = 0;
    bool tied_lm_head = false;

    int q_dim() const { return q_heads * head_dim; }
    int kv_dim() const { return kv_heads * head_dim; }
    int lin_qk_dim() const { return lin_k_heads * lin_k_dim; }
    int lin_value_dim() const { return lin_v_heads * lin_v_dim; }
    int conv_channels() const { return 2 * lin_qk_dim() + lin_value_dim(); }
    bool is_full(int layer) const { return (layer + 1) % full_every == 0; }
};

/// \brief group-64 ONNX MatMulNBits components for one biased projection
struct OnnxQ8 {
    std::vector<std::uint8_t> qweight;
    std::vector<std::uint16_t> scales;
    std::vector<std::uint8_t> qzeros;
};

class Qwen35Gguf {
public:
    static std::shared_ptr<Qwen35Gguf> Open(const std::filesystem::path& path);

    const std::filesystem::path& Path() const { return path_; }
    const Qwen35Config& Config() const { return config_; }
    const std::string& LmHeadName() const { return lm_head_; }
    const rai::GgufTensor& Embedding() const { return embedding_; }

    /// \brief raw Q8_0 bytes for an [n, k] matrix, k fastest
    std::span<const std::byte> Blocks(std::string_view base, std::int64_t k, std::int64_t n) const;
    std::span<const float> F32(std::string_view name, std::span<const std::int64_t> shape) const;

    std::vector<std::byte> QkvBlocks(int layer) const;
    std::vector<std::byte> ZBlocks(int layer) const;
    std::vector<std::byte> GateBlocks(int layer, std::string_view which) const;
    std::vector<std::byte> OutBlocks(int layer) const;
    std::vector<float> PerHead(int layer, std::string_view name) const;
    /// \brief [channels, width] float32, channels already in HF order
    std::vector<float> ConvTaps(int layer) const;

private:
    Qwen35Gguf(std::shared_ptr<rai::GgufFile> file, std::filesystem::path path,
               Qwen35Config config, std::string lm_head, rai::GgufTensor embedding,
               std::vector<std::int64_t> order);

    std::vector<std::byte> TakeRows(std::span<const std::byte> raw, std::int64_t n, std::int64_t k,
                                    std::span<const std::int64_t> rows) const;
    std::vector<std::int64_t> ChannelOrder() const;

    std::shared_ptr<rai::GgufFile> file_;
    std::filesystem::path path_;
    Qwen35Config config_;
    std::string lm_head_;
    rai::GgufTensor embedding_;
    std::vector<std::int64_t> order_;
};

/// \brief refit Q8_0 [n, k] into the ONNX layout from_onnx packs, group 64
/// \note Byte-identical to gguf_requantized where both apply. Used only for the
///       alpha gate, which needs a bias the GGUF requantizer has no slot for.
OnnxQ8 Q8ToOnnx(std::span<const std::byte> blocks, int k, int n, int group);

}  // namespace flm::qwen35
