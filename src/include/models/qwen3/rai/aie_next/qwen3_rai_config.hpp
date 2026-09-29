/// \file qwen3_rai_config.hpp
/// \brief The Qwen3 sizes the rai engine runs, and the kernel grid it runs on
/// \note One engine serves four sizes, so the architecture is a row of a table
///       rather than a block of constants. The GGUF's own metadata selects the
///       row and is cross-checked against it; the row supplies what the file
///       cannot state -- the group sizes, which are properties of which ELFs
///       ship, not of the model.
#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <string_view>

namespace flm::qwen3 {

struct Qwen3Config {
    std::string_view size;
    std::int64_t layers;
    std::int64_t hidden;
    std::int64_t q_heads;
    std::int64_t kv_heads;
    std::int64_t head_dim;
    std::int64_t intermediate;
    std::int64_t vocab;
    double rope_theta;
    float epsilon;
    /// \brief the requantized group of every projection and the MLP
    std::uint32_t group;
    /// \brief the group lm_head packs at, which is a different ELF family
    std::uint32_t head_group;

    std::int64_t q_dim() const noexcept { return q_heads * head_dim; }
    std::int64_t kv_dim() const noexcept { return kv_heads * head_dim; }
};

/// \brief the four sizes ryzenai-corelib's qwen3 driver runs on the NPU
/// \note Transcribed from corelib's QWEN table (python/model_driver/
///       qwen3_driver.py), which is where the claim that they run is kept.
///       0.6B is the only row whose lm_head packs at group 32: 1024 -> 151936
///       ships a group-32 kernel only, while everything else in that model
///       ships group 64 only.
inline constexpr std::array<Qwen3Config, 4> kQwen3Rows{{
    {"0.6B", 28, 1024, 16, 8, 128, 3072, 151936, 1.0e6, 1.0e-6f, 64, 32},
    {"1.7B", 28, 2048, 16, 8, 128, 6144, 151936, 1.0e6, 1.0e-6f, 64, 64},
    {"4B", 36, 2560, 32, 8, 128, 9728, 151936, 1.0e6, 1.0e-6f, 64, 64},
    {"8B", 36, 4096, 32, 8, 128, 12288, 151936, 1.0e6, 1.0e-6f, 64, 64},
}};

/// \brief the arch fields a GGUF states, as read from its metadata
struct Qwen3Shape {
    std::int64_t layers, hidden, q_heads, kv_heads, head_dim, intermediate;
    double rope_theta;
    double epsilon;
};

/// \brief the row a file's shape selects
/// \throws std::runtime_error when no row matches, or when the matching row
///         disagrees with the file about epsilon or rope theta (Qwen3-4B-2507
///         is shape-identical to Qwen3-4B but ropes at 5e6, not 1e6)
const Qwen3Config& SelectQwen3Row(const Qwen3Shape& shape);

/// \brief rows buffers and the attention ELF are built for; a power of two
inline constexpr std::int64_t kMaxSequenceLength = 4096;
/// \brief the last position a decode may write
inline constexpr std::int64_t kMaxDecodeWindow = 4095;

/// \brief the row counts every op these models dispatch ships a kernel for
/// \note matmul, ssmlp and flat_mha share this grid at every covered width,
///       which is what lets one padded row count serve a whole pass. The shape
///       plan asks corelib and refuses to load if any op disagrees.
inline constexpr std::array<std::int64_t, 9> kExecutionRows{
    1, 64, 128, 256, 512, 1024, 2048, 3072, 4096};

/// \brief QK-Norm's width, and every M a k=128 RMSNorm kernel ships at
/// \note Read off dyn_bins.dll's RMSNorm kernels (token PDI and prefill PDI):
///       8/16/32 on the token PDI, 512 and up on prefill, and 33..511 a hole
///       no PDI covers. corelib's rmsnorm pad helper is an identity and cannot
///       report this, so the shape plan checks rows x heads against it.
inline constexpr std::int64_t kQkNormWidth = 128;
inline constexpr std::array<std::int64_t, 15> kQkNormRows{
    8, 16, 32, 512, 1024, 2048, 4096, 8192, 16384, 24576, 32768, 49152,
    65536, 98304, 131072};

/// \brief the PDI pair every Qwen3 shape ships under (corelib's PDI_PAIR)
inline constexpr int kPrefillPdi = 1;
inline constexpr int kTokenPdi = 16;

/// \brief intra-packer thread hint; corelib treats 0 as one
inline constexpr std::uint32_t kRequantizeThreads = 0;

/// \brief how many requantizing creates run at once
/// \note Serial, as corelib's qwen3 driver loads. corelib records an open,
///       unattributed defect on both requantizing entry points in which
///       concurrent creates correlate with all-zero device output; serial is
///       the configuration it has not been observed in. The weight cache makes
///       this a first-load cost only.
inline constexpr std::size_t kWeightCreateConcurrency = 1;

/// \brief Qwen3's two end ids: <|im_end|> ends a turn, <|endoftext|> a document
/// \note The GGUF and tokenizer_config.json each state only 151645; the pair is
///       Qwen's generation_config.json. Both are checked against tokenizer.json.
inline constexpr std::array<int, 2> kEosIds{151645, 151643};

}  // namespace flm::qwen3
