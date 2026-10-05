#include "models/gemma4/rai/aie_next/gemma4_rai_host.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstring>
#include <exception>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#if defined(_M_X64) || defined(__x86_64__)
#define FLM_GEMMA4_X86 1
#include <immintrin.h>
#if defined(_MSC_VER)
#include <intrin.h>
#define FLM_GEMMA4_AVX512
#else
#include <cpuid.h>
#define FLM_GEMMA4_AVX512 __attribute__((target("avx512f,avx2")))
#endif
#endif

namespace flm::gemma4 {
namespace {

// IEEE binary16 -> float32. Q8_0 blocks store their scale as one fp16 value;
// this mirrors the bit layout phi4_rai_host.cpp already decodes Q8_0 with,
// so the two model ports agree on what a Q8_0 block means.
float HalfToFloat(std::uint16_t half) {
    const std::uint32_t sign = static_cast<std::uint32_t>(half & 0x8000) << 16;
    const std::uint32_t exponent = (half >> 10) & 0x1f;
    std::uint32_t fraction = half & 0x03ff;
    std::uint32_t bits;
    if (exponent == 0) {
        if (fraction == 0) {
            bits = sign;
        } else {
            int shift = 0;
            while ((fraction & 0x0400) == 0) {
                fraction <<= 1;
                ++shift;
            }
            fraction &= 0x03ff;
            bits = sign | (static_cast<std::uint32_t>(127 - 14 - shift) << 23) |
                   (fraction << 13);
        }
    } else if (exponent == 0x1f) {
        bits = sign | 0x7f800000 | (fraction << 13);
    } else {
        bits = sign | ((exponent + (127 - 15)) << 23) | (fraction << 13);
    }
    return std::bit_cast<float>(bits);
}

// THE PROJECTION: out[r, c] = sum_i x[r, i] * bf16(w[c, i]) for columns
// [c_begin, c_end), accumulated in DOUBLE. The reference comparison in
// test_gemma4_real_gguf.cpp holds these planes to 1e-6 absolute, and FP32
// accumulation over hidden = 2560 is itself ~1e-6 -- so the precision stays and
// the speed comes from blocking, SIMD and threads.
constexpr std::size_t kProjectColumnTile = 64;
constexpr std::size_t kProjectRowBlock = 16;

void ProjectScalar(const float* x, const std::uint16_t* w, float* out, std::size_t rows,
                   std::size_t c_begin, std::size_t c_end, std::size_t width,
                   std::size_t hidden) {
    for (std::size_t row = 0; row < rows; ++row) {
        const float* x_row = x + row * hidden;
        for (std::size_t column = c_begin; column < c_end; ++column) {
            const std::uint16_t* weights = w + column * hidden;
            double sum = 0.0;
            for (std::size_t i = 0; i < hidden; ++i) {
                // BF16 widens EXACTLY: it is the top 16 bits of a binary32.
                const auto bits = static_cast<std::uint32_t>(weights[i]) << 16;
                sum += static_cast<double>(x_row[i]) *
                       static_cast<double>(std::bit_cast<float>(bits));
            }
            out[row * width + column] = static_cast<float>(sum);
        }
    }
}

#if defined(FLM_GEMMA4_X86)
bool CpuHasAvx512F() {
    int regs[4] = {};
#if defined(_MSC_VER)
    __cpuid(regs, 0);
    if (regs[0] < 7) return false;
    __cpuid(regs, 1);
    const bool osxsave = (regs[2] >> 27) & 1;
    __cpuidex(regs, 7, 0);
    const bool avx512f = (regs[1] >> 16) & 1, avx2 = (regs[1] >> 5) & 1;
    if (!osxsave || !avx512f || !avx2) return false;
    const auto xcr0 = _xgetbv(0);
#else
    unsigned a, b, c, d;
    if (!__get_cpuid_count(7, 0, &a, &b, &c, &d)) return false;
    const bool avx512f = (b >> 16) & 1, avx2 = (b >> 5) & 1;
    if (!__get_cpuid(1, &a, &b, &c, &d)) return false;
    const bool osxsave = (c >> 27) & 1;
    if (!osxsave || !avx512f || !avx2) return false;
    unsigned lo, hi;
    __asm__("xgetbv" : "=a"(lo), "=d"(hi) : "c"(0));
    const unsigned long long xcr0 = (static_cast<unsigned long long>(hi) << 32) | lo;
#endif
    (void)regs;
    // XMM, YMM, opmask, ZMM_Hi256 and Hi16_ZMM state all enabled by the OS.
    return (xcr0 & 0xE6) == 0xE6;
}

// NR rows x 4 columns, eight doubles per lane. Columns past `columns` repeat
// the last one and are not written.
template <int NR>
FLM_GEMMA4_AVX512 void ProjectTileAvx512(const double* x, const std::uint16_t* w,
                                        float* out, std::size_t column,
                                        std::size_t columns, std::size_t width,
                                        std::size_t hidden) {
    __m512d acc[NR][4];
    for (auto& row : acc)
        for (auto& lane : row) lane = _mm512_setzero_pd();
    const std::uint16_t* wp[4];
    for (std::size_t k = 0; k < 4; ++k)
        wp[k] = w + (column + std::min(k, columns - 1)) * hidden;
    const std::size_t body = hidden & ~std::size_t{7};
    for (std::size_t i = 0; i < body; i += 8) {
        __m512d wv[4];
        for (std::size_t k = 0; k < 4; ++k) {
            const __m128i h = _mm_loadu_si128(reinterpret_cast<const __m128i*>(wp[k] + i));
            wv[k] = _mm512_cvtps_pd(
                _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_cvtepu16_epi32(h), 16)));
        }
        for (int j = 0; j < NR; ++j) {
            const __m512d xv = _mm512_loadu_pd(x + static_cast<std::size_t>(j) * hidden + i);
            for (std::size_t k = 0; k < 4; ++k) acc[j][k] = _mm512_fmadd_pd(xv, wv[k], acc[j][k]);
        }
    }
    for (int j = 0; j < NR; ++j) {
        const double* x_row = x + static_cast<std::size_t>(j) * hidden;
        for (std::size_t k = 0; k < columns; ++k) {
            double sum = _mm512_reduce_add_pd(acc[j][k]);
            for (std::size_t i = body; i < hidden; ++i) {
                const auto bits = static_cast<std::uint32_t>(wp[k][i]) << 16;
                sum += x_row[i] * static_cast<double>(std::bit_cast<float>(bits));
            }
            out[static_cast<std::size_t>(j) * width + column + k] = static_cast<float>(sum);
        }
    }
}

FLM_GEMMA4_AVX512 void ProjectAvx512(const float* x, const std::uint16_t* w, float* out,
                                     std::size_t rows, std::size_t c_begin,
                                     std::size_t c_end, std::size_t width,
                                     std::size_t hidden, std::vector<double>& block) {
    block.resize(kProjectRowBlock * hidden);
    for (std::size_t r0 = 0; r0 < rows; r0 += kProjectRowBlock) {
        const std::size_t nr = std::min(kProjectRowBlock, rows - r0);
        for (std::size_t i = 0; i < nr * hidden; ++i) block[i] = x[r0 * hidden + i];
        for (std::size_t c = c_begin; c < c_end; c += 4) {
            const std::size_t nc = std::min<std::size_t>(4, c_end - c);
            std::size_t r = 0;
            float* o = out + r0 * width;
            for (; r + 4 <= nr; r += 4)
                ProjectTileAvx512<4>(block.data() + r * hidden, w, o + r * width, c, nc, width, hidden);
            const double* xr = block.data() + r * hidden;
            switch (nr - r) {
                case 3: ProjectTileAvx512<3>(xr, w, o + r * width, c, nc, width, hidden); break;
                case 2: ProjectTileAvx512<2>(xr, w, o + r * width, c, nc, width, hidden); break;
                case 1: ProjectTileAvx512<1>(xr, w, o + r * width, c, nc, width, hidden); break;
                default: break;
            }
        }
    }
}
#endif

// Column tiles are dealt round-robin to one thread per physical core (the
// weights, 55 MB on E4B, are read once per call either way).
void Project(const float* x, const std::uint16_t* w, float* out, std::size_t rows,
             std::size_t width, std::size_t hidden) {
#if defined(FLM_GEMMA4_X86)
    static const bool avx512 = CpuHasAvx512F();
#else
    constexpr bool avx512 = false;
#endif
    const std::size_t tiles = (width + kProjectColumnTile - 1) / kProjectColumnTile;
    const std::size_t cores = std::max(1u, std::thread::hardware_concurrency() / 2);
    const std::size_t workers = std::min(tiles, cores);

    std::exception_ptr failure;
    std::mutex failure_mutex;
    const auto run = [&](std::size_t worker) {
        try {
            std::vector<double> block;
            for (std::size_t tile = worker; tile < tiles; tile += workers) {
                const std::size_t c_begin = tile * kProjectColumnTile;
                const std::size_t c_end = std::min(width, c_begin + kProjectColumnTile);
#if defined(FLM_GEMMA4_X86)
                if (avx512) {
                    ProjectAvx512(x, w, out, rows, c_begin, c_end, width, hidden, block);
                    continue;
                }
#endif
                ProjectScalar(x, w, out, rows, c_begin, c_end, width, hidden);
            }
        } catch (...) {
            std::lock_guard<std::mutex> lock(failure_mutex);
            if (!failure) failure = std::current_exception();
        }
    };
    std::vector<std::thread> pool;
    pool.reserve(workers - 1);
    try {
        for (std::size_t worker = 1; worker < workers; ++worker) pool.emplace_back(run, worker);
    } catch (...) {
        for (auto& thread : pool) thread.join();
        throw;
    }
    run(0);
    for (auto& thread : pool) thread.join();
    if (failure) std::rethrow_exception(failure);
}

}  // namespace

RopeTables MakeRopeTables(std::int64_t head_dim, double theta,
                          std::int64_t positions,
                          std::span<const float> frequency_factors) {
    if (head_dim <= 0 || head_dim % 2 != 0) {
        throw std::runtime_error("rope head_dim must be positive and even, got " +
                                 std::to_string(head_dim));
    }
    const std::int64_t half = head_dim / 2;
    // ONE FACTOR PER ROTARY PAIR. The driver checks the same thing and says
    // the same thing: "rope frequency factors are (n,), not (half,) -- one per
    // rotary pair of a `rope_dim`-wide head". A vector of another length is a
    // file this code has not been read against, not a value to stretch.
    if (!frequency_factors.empty() &&
        frequency_factors.size() != static_cast<std::size_t>(half)) {
        throw std::runtime_error(
            "rope frequency factors are " + std::to_string(frequency_factors.size()) +
            " long, not " + std::to_string(half) +
            " -- one per rotary pair of a " + std::to_string(head_dim) + "-wide head");
    }
    RopeTables tables;
    tables.head_dim = head_dim;
    tables.positions = positions;
    tables.cos.resize(static_cast<std::size_t>(positions * half));
    tables.sin.resize(static_cast<std::size_t>(positions * half));
    for (std::int64_t pos = 0; pos < positions; ++pos) {
        for (std::int64_t i = 0; i < half; ++i) {
            double freq = 1.0 / std::pow(theta,
                (2.0 * static_cast<double>(i)) / static_cast<double>(head_dim));
            // DIVIDES. See the header: this is `theta /= freq_factors[i/2]` in
            // llama.cpp and `freq = freq / factors` in the reference driver.
            // Multiplying instead produces a table that is finite, bounded by
            // 1, indistinguishable by any magnitude check, and wrong from
            // position 1 on.
            if (!frequency_factors.empty())
                freq /= static_cast<double>(frequency_factors[static_cast<std::size_t>(i)]);
            const double angle = static_cast<double>(pos) * freq;
            const auto index = static_cast<std::size_t>(pos * half + i);
            tables.cos[index] = static_cast<float>(std::cos(angle));
            tables.sin[index] = static_cast<float>(std::sin(angle));
        }
    }
    return tables;
}

void RmsNorm(std::span<const float> x, std::span<const float> gamma, float epsilon,
             std::span<float> out) {
    if (gamma.size() != x.size()) {
        throw std::runtime_error("RmsNorm gamma width " + std::to_string(gamma.size()) +
                                  " does not match x width " + std::to_string(x.size()));
    }
    if (out.size() != x.size()) {
        throw std::runtime_error("RmsNorm out width " + std::to_string(out.size()) +
                                  " does not match x width " + std::to_string(x.size()));
    }
    double sum_of_squares = 0.0;
    for (const float value : x) sum_of_squares += double(value) * value;
    const double mean_square = sum_of_squares / static_cast<double>(x.size());
    const double scale = 1.0 / std::sqrt(mean_square + static_cast<double>(epsilon));
    for (std::size_t i = 0; i < x.size(); ++i) {
        out[i] = static_cast<float>(static_cast<double>(x[i]) * scale *
                                     static_cast<double>(gamma[i]));
    }
}

void DecodeQ8Row(std::span<const std::byte> row, std::span<float> out) {
    constexpr std::size_t kBlockBytes = 34;   // 2-byte fp16 scale + 32 int8 codes.
    constexpr std::size_t kBlockElements = 32;
    if (row.size() % kBlockBytes != 0) {
        throw std::runtime_error("Q8_0 row byte length " + std::to_string(row.size()) +
                                  " is not a multiple of " + std::to_string(kBlockBytes));
    }
    const std::size_t blocks = row.size() / kBlockBytes;
    const std::size_t expected_out = blocks * kBlockElements;
    if (out.size() != expected_out) {
        throw std::runtime_error("Q8_0 decode out width " + std::to_string(out.size()) +
                                  " does not match expected width " +
                                  std::to_string(expected_out));
    }
    for (std::size_t block = 0; block < blocks; ++block) {
        const std::byte* encoded = row.data() + block * kBlockBytes;
        std::uint16_t scale_bits;
        std::memcpy(&scale_bits, encoded, sizeof(scale_bits));
        const float scale = HalfToFloat(scale_bits);
        for (std::size_t element = 0; element < kBlockElements; ++element) {
            const auto code = static_cast<std::int8_t>(
                std::to_integer<std::uint8_t>(encoded[2 + element]));
            out[block * kBlockElements + element] = scale * static_cast<float>(code);
        }
    }
}

void FloatsToBf16(std::span<const float> values, std::span<std::uint16_t> out) {
    if (out.size() != values.size()) {
        throw std::runtime_error("FloatsToBf16 out width " + std::to_string(out.size()) +
                                  " does not match values width " +
                                  std::to_string(values.size()));
    }
    for (std::size_t i = 0; i < values.size(); ++i) {
        std::uint32_t bits = std::bit_cast<std::uint32_t>(values[i]);
        bits += 0x7FFF + ((bits >> 16) & 1);
        out[i] = static_cast<std::uint16_t>(bits >> 16);
    }
}

void GatherPerLayerEmbedding(std::span<const std::byte> per_layer_token_embd,
                             std::int64_t layers, std::int64_t ple_dim,
                             std::int64_t vocab, std::int64_t token,
                             std::span<float> out) {
    if (layers <= 0 || ple_dim <= 0 || vocab <= 0) {
        throw std::runtime_error(
            "GatherPerLayerEmbedding layers, ple_dim and vocab must be positive");
    }
    if (token < 0 || token >= vocab) {
        throw std::runtime_error("token " + std::to_string(token) +
                                  " is outside vocabulary of size " + std::to_string(vocab));
    }
    const std::int64_t rows = layers * ple_dim;
    if (out.size() != static_cast<std::size_t>(rows)) {
        throw std::runtime_error("GatherPerLayerEmbedding out width " +
                                  std::to_string(out.size()) +
                                  " does not match layers * ple_dim " +
                                  std::to_string(rows));
    }

    // THE MEMORY LAYOUT, AND IT IS THE OPPOSITE OF WHAT THIS USED TO ASSUME.
    //
    // `per_layer_token_embd.weight`'s GGUF dims are
    // `[layers * ple_dim, vocab]`, and dims[0] is the FASTEST-VARYING extent
    // -- so the bytes are `vocab` rows of `layers * ple_dim` contiguous
    // elements, with the Q8_0 blocks running along the PLE axis. One token's
    // whole per-layer slice is ONE CONTIGUOUS ROW: 280 blocks on E2B, 336 on
    // E4B. That is exactly what the reference driver does
    // (`gemma4_driver.py`'s `per_layer_embedding()` builds
    // `LazyEmbedding(data, dtype, (dims[1], dims[0]))` and slices row i at
    // `i * (nbytes // vocab)`), and what `_ple_inputs` then reshapes to
    // `[rows, layers, ple_dim]`.
    //
    // This function previously read the transpose -- `layers * ple_dim` rows
    // of `vocab`, one block plucked from each -- because
    // verified-gguf-facts.md's "(layers * ple_dim, vocab)" was taken as a
    // row-major shape rather than as GGUF dims, and that document said in so
    // many words that no change was needed here. THE TWO LAYOUTS HAVE THE
    // IDENTICAL TOTAL BYTE COUNT (8960 x 262144 elements either way), so
    // every length check passed and the result was 8960 finite, plausible,
    // wrong floats. test_gemma4_real_gguf.cpp now pins it against values
    // dumped from the reference driver's own rule over the real files.
    constexpr std::int64_t kBlockElements = 32;
    constexpr std::size_t kBlockBytes = 34;
    if (rows % kBlockElements != 0) {
        throw std::runtime_error(
            "layers * ple_dim is " + std::to_string(rows) +
            ", which is not a multiple of 32; the Q8_0 blocks run along THIS "
            "axis, not along the vocabulary");
    }
    const std::size_t row_bytes =
        static_cast<std::size_t>(rows / kBlockElements) * kBlockBytes;
    const std::size_t required_bytes = static_cast<std::size_t>(vocab) * row_bytes;
    if (per_layer_token_embd.size() < required_bytes) {
        throw std::runtime_error("per_layer_token_embd byte length " +
                                  std::to_string(per_layer_token_embd.size()) +
                                  " is smaller than the " + std::to_string(required_bytes) +
                                  " required for " + std::to_string(vocab) +
                                  " rows of " + std::to_string(rows));
    }

    const auto row = per_layer_token_embd.subspan(
        static_cast<std::size_t>(token) * row_bytes, row_bytes);
    DecodeQ8Row(row, out);
}

std::vector<std::vector<float>> BuildPerLayerInputs(
    std::span<const std::byte> per_layer_token_embd,
    std::span<const std::uint16_t> model_proj,
    std::span<const float> proj_norm_gamma,
    std::span<const float> scaled_embedding, std::span<const int> ids,
    std::int64_t layers, std::int64_t ple_dim, std::int64_t hidden,
    std::int64_t vocab, float epsilon) {
    if (layers <= 0 || ple_dim <= 0 || hidden <= 0 || vocab <= 0) {
        throw std::runtime_error(
            "BuildPerLayerInputs layers, ple_dim, hidden and vocab must be positive");
    }
    const auto rows = ids.size();
    if (rows == 0) throw std::runtime_error("BuildPerLayerInputs was given no token ids");
    const auto width = static_cast<std::size_t>(layers * ple_dim);
    if (proj_norm_gamma.size() != static_cast<std::size_t>(ple_dim)) {
        throw std::runtime_error("per_layer_proj_norm is " +
                                 std::to_string(proj_norm_gamma.size()) +
                                 " wide, not ple_dim " + std::to_string(ple_dim));
    }
    if (model_proj.size() != width * static_cast<std::size_t>(hidden)) {
        throw std::runtime_error("per_layer_model_proj holds " +
                                 std::to_string(model_proj.size()) +
                                 " elements, not layers * ple_dim * hidden " +
                                 std::to_string(width * static_cast<std::size_t>(hidden)));
    }
    if (scaled_embedding.size() != rows * static_cast<std::size_t>(hidden)) {
        throw std::runtime_error("the scaled embedding holds " +
                                 std::to_string(scaled_embedding.size()) +
                                 " elements, not rows * hidden " +
                                 std::to_string(rows * static_cast<std::size_t>(hidden)));
    }

    const auto ple_scale = static_cast<float>(std::sqrt(static_cast<double>(ple_dim)));
    const auto projection_scale =
        static_cast<float>(1.0 / std::sqrt(static_cast<double>(hidden)));
    const auto sum_scale = static_cast<float>(1.0 / std::sqrt(2.0));

    std::vector<std::vector<float>> planes(
        static_cast<std::size_t>(layers),
        std::vector<float>(rows * static_cast<std::size_t>(ple_dim), 0.0f));

    // Projected a chunk of rows at a time, so a 4096-token prompt does not hold
    // a second [rows, width] FP32 copy beside `planes`.
    constexpr std::size_t kRowChunk = 256;
    const auto hidden_size = static_cast<std::size_t>(hidden);
    std::vector<float> gathered(width);
    std::vector<float> projected(std::min(rows, kRowChunk) * width);
    std::vector<float> normed(static_cast<std::size_t>(ple_dim));
    for (std::size_t chunk = 0; chunk < rows; chunk += kRowChunk) {
        const std::size_t chunk_rows = std::min(kRowChunk, rows - chunk);
        // THE PROJECTION, AGAINST THE TRANSPOSE OF THE MAPPING. Output column
        // `c` is the dot product of a row's embedding with mapping ROW `c`
        // -- `per_layer_model_proj.weight`'s file dims are [hidden, width] with
        // the input width fastest-varying, so the memory is `width` rows of
        // `hidden`, and the driver's `embed_rows(ids) @ floats(...)` is against
        // `floats()`'s transposed view of it. Reading it the other way round
        // has the identical element count.
        Project(scaled_embedding.data() + chunk * hidden_size, model_proj.data(),
                projected.data(), chunk_rows, width, hidden_size);

        for (std::size_t local = 0; local < chunk_rows; ++local) {
            const std::size_t row = chunk + local;
            const auto token = static_cast<std::int64_t>(ids[row]);
            // ggml_get_rows, then ggml_scale(sqrt(n_embd_per_layer)). The gather
            // reads ONE CONTIGUOUS ROW of the Q8_0 table; see
            // GatherPerLayerEmbedding for why the transpose has the identical
            // byte count and returns plausible nonsense.
            GatherPerLayerEmbedding(per_layer_token_embd, layers, ple_dim, vocab, token,
                                    gathered);
            float* projected_row = projected.data() + local * width;
            for (std::size_t column = 0; column < width; ++column)
                projected_row[column] *= projection_scale;

            for (std::int64_t layer = 0; layer < layers; ++layer) {
                const auto offset = static_cast<std::size_t>(layer * ple_dim);
                // PER LAYER SLICE, not over the whole width: each layer's
                // `ple_dim` values are normalized on their own, with the one
                // `per_layer_proj_norm` gamma.
                RmsNorm(std::span<const float>(projected_row + offset,
                                               static_cast<std::size_t>(ple_dim)),
                        proj_norm_gamma, epsilon, normed);
                auto& plane = planes[static_cast<std::size_t>(layer)];
                for (std::int64_t i = 0; i < ple_dim; ++i) {
                    const auto index = static_cast<std::size_t>(i);
                    plane[row * static_cast<std::size_t>(ple_dim) + index] =
                        (normed[index] + gathered[offset + index] * ple_scale) * sum_scale;
                }
            }
        }
    }
    return planes;
}

void SoftcapLogits(std::span<float> logits, float cap) {
    if (!(cap > 0.0f)) {
        throw std::runtime_error("logit softcap must be positive, got " +
                                  std::to_string(cap));
    }
    for (float& logit : logits) {
        logit = std::tanh(logit / cap) * cap;
    }
}

}  // namespace flm::gemma4
