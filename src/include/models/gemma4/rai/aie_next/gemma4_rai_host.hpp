#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace flm::gemma4 {

struct RopeTables {
    std::vector<float> cos;   // [positions * (head_dim / 2)]
    std::vector<float> sin;
    std::int64_t head_dim;
    std::int64_t positions;
};

/// \brief the rotary tables for one geometry
/// \param frequency_factors `rope_freqs.weight`, one entry per rotary PAIR,
///        or empty for a regime that has none
/// \note Gemma 4 has TWO thetas -- 1e6 on full-attention layers and 1e4 on
///       sliding ones -- so this is called twice and the results are not
///       interchangeable. RoPE frequencies apply from position 0 upward, so a
///       wrong theta is wrong at position 1: the 4096 context cap does not
///       rescue it.
///
/// \note THE FACTORS DIVIDE, AND THEY BELONG TO THE FULL-ATTENTION REGIME
///       ALONE. `config.json`'s `partial_rotary_factor: 0.25` is NOT a shorter
///       rotary -- `rope_dim == head_size` on both geometries, confirmed by
///       `gemma4.rope.dimension_count` (512) and `..._swa` (256) on both
///       shipped rows. It is a per-pair FREQUENCY SCALING carried in
///       `rope_freqs.weight`, which is why that tensor is [256] on both
///       models: one entry per pair of a 512-wide head. llama.cpp passes it as
///       `freq_factors` on full-attention layers only, where it DIVIDES
///       (`theta /= freq_factors[i/2]`), and the reference driver writes
///       `freq = freq / factors` (`gemma4_driver.py` `_rope_tables`).
///       Sliding layers pass none.
///
/// \note WHICH DIRECTION THE DIVISION RUNS IS NOT VISIBLE IN ANY MAGNITUDE
///       ASSERTION -- both tables are bounded by 1 and both look like rotary
///       tables -- so it is pinned by the llama.cpp/driver comparison and, in
///       this suite, by the position-doubling identity in
///       TestRopeFrequencyFactorsDivideTheFrequency.
///
/// \throws std::runtime_error naming both lengths, if `frequency_factors` is
///         non-empty and is not exactly `head_dim / 2` long
RopeTables MakeRopeTables(std::int64_t head_dim, double theta,
                          std::int64_t positions,
                          std::span<const float> frequency_factors = {});

/// \brief scale * x / sqrt(mean(x^2) + epsilon)
void RmsNorm(std::span<const float> x, std::span<const float> gamma, float epsilon,
             std::span<float> out);

/// \brief decode one Q8_0 row into floats
/// \note Q8_0 is blocks of 32 int8 codes with one fp16 scale each.
void DecodeQ8Row(std::span<const std::byte> row, std::span<float> out);

/// \brief round-to-nearest-even f32 -> bf16, as the kernels expect
void FloatsToBf16(std::span<const float> values, std::span<std::uint16_t> out);

/// \brief gather one token's per-layer embedding slice for every layer
/// \param per_layer_token_embd the mapped Q8_0 tensor, GGUF dims
///        `[layers * ple_dim, vocab]`
/// \param token the token id
/// \param out [layers * ple_dim] floats
/// \note per_layer_token_embd's dims are (8960, 262144) on E2B -- 2.35
///       BILLION parameters in one tensor, which is why a model named E2B has
///       a 5.05 GB Q8_0 file, larger than Q8_0 Qwen3-8B. It stays MAPPED and
///       is gathered per token; materializing it is not an option at any
///       point.
/// \note THE LAYOUT IS `vocab` ROWS OF `layers * ple_dim`, not the transpose.
///       GGUF dims run fastest-varying first, so dims[0] -- the PLE axis --
///       is the contiguous one and a token's whole slice is ONE ROW of 280
///       Q8_0 blocks (E2B) or 336 (E4B). This is what the reference driver
///       does (`gemma4_driver.py`'s `per_layer_embedding()` /
///       `LazyEmbedding.rows`).
///
///       Read the other way round -- as `layers * ple_dim` rows of `vocab`,
///       which is what this function did until Task R1 -- the TOTAL BYTE
///       COUNT IS IDENTICAL, so nothing rejects it and the result is 8960
///       finite, plausible, wrong floats. There is no shape check anywhere
///       that can catch this; only comparing values against the reference
///       over a real file does, which test_gemma4_real_gguf.cpp now does.
/// \throws std::runtime_error if `layers * ple_dim` is not a multiple of 32
///         (the Q8_0 block size along the contiguous axis), if `token` is
///         outside the vocabulary, if `out` is not `layers * ple_dim` wide,
///         or if the mapped tensor is shorter than `vocab` such rows.
void GatherPerLayerEmbedding(std::span<const std::byte> per_layer_token_embd,
                             std::int64_t layers, std::int64_t ple_dim,
                             std::int64_t vocab, std::int64_t token,
                             std::span<float> out);

/// \brief the per-layer embedding input planes for a prompt, layer-major
///
/// \param per_layer_token_embd the mapped Q8_0 `per_layer_token_embd.weight`
/// \param model_proj the mapped BF16 `per_layer_model_proj.weight`, which is
///        `layers * ple_dim` rows of `hidden` in memory
/// \param proj_norm_gamma `per_layer_proj_norm.weight`, `ple_dim` wide
/// \param scaled_embedding `[rows, hidden]`, the token embedding rows ALREADY
///        multiplied by `sqrt(hidden)`
/// \param ids the prompt, `rows` of them
/// \return `layers` planes, each `rows * ple_dim` floats
///
/// THE ONE THING A FORWARD PASS COMPUTES ON THE HOST, and deliberately so, for
/// two independent reasons the reference driver gives (`_ple_inputs`).
/// `per_layer_model_proj` is `hidden -> layers * ple_dim` and is UNREACHABLE at
/// this model's (8, 17) PDI pair on both sizes -- E2B's 1536 -> 8960 ships at
/// group 64 only and entirely off-pair, E4B's 2560 -> 10752 at no pair and no
/// group at all -- so there is no ELF for a stream to dispatch. And llama.cpp
/// puts this same matmul on the CPU regardless: `project_per_layer_inputs`
/// says "this matrix multiplication will be performed in the input layer (i.e.
/// on the CPU)". This is not a placeholder for an NPU path.
///
/// IT IS A GEMM, AND THE PREFILL's LARGEST HOST COST: `hidden * layers *
/// ple_dim` multiply-adds per row, 27.5 million on E4B. Keep it blocked, SIMD
/// and threaded: as one scalar dot product per (row, column) it measured ~16 ms
/// a row on the bring-up machine -- ~35 s of a 2,159-token TTFT, and ~16 ms of every
/// decode step. It ACCUMULATES IN DOUBLE on purpose: FP32 would halve the time
/// but is itself ~1e-6 over hidden = 2560, the tolerance the real-file
/// reference test holds these planes to.
///
/// FOUR SCALE FACTORS, AND NONE OF THEM CAN MAKE THE OUTPUT NON-FINITE, so a
/// missing one is a wrong magnitude and fluent text:
///
///   - `sqrt(ple_dim)` on the gathered per-layer embedding;
///   - `1/sqrt(hidden)` on the projection;
///   - `1/sqrt(2)` on the sum of the two;
///   - `sqrt(hidden)` on the token embedding, which belongs to the CALLER
///     (`scaled_embedding` is already scaled) because llama.cpp scales `inpL`
///     once, immediately after the gather. Applying it here as well multiplies
///     the projection by 39 and still produces finite, fluent-looking output.
///
/// THE RMSNorm DOES NOT DIVIDE `1/sqrt(hidden)` BACK OUT, which is why it is
/// not redundant: `rms_norm(c*x) = x / sqrt(mean(x^2) + eps/c^2)`, so a uniform
/// scale RESCALES EPSILON. On this model eps is 1e-6 against a pre-norm
/// `mean(proj^2)` of 3.3e-3..7.4e-2, so the normalised rows come out at RMS
/// 0.999848 rather than 1; dropping the factor takes that to 0.999998. A small
/// effect, not an absent one.
///
/// LAYER-MAJOR, so one layer's `[rows, ple_dim]` slice is contiguous and is a
/// single write into that layer's own device tensor.
///
/// \throws std::runtime_error naming the offending length, if any span is not
///         the size the other arguments imply, or if a token id is outside the
///         vocabulary
std::vector<std::vector<float>> BuildPerLayerInputs(
    std::span<const std::byte> per_layer_token_embd,
    std::span<const std::uint16_t> model_proj,
    std::span<const float> proj_norm_gamma,
    std::span<const float> scaled_embedding, std::span<const int> ids,
    std::int64_t layers, std::int64_t ple_dim, std::int64_t hidden,
    std::int64_t vocab, float epsilon);

/// \brief tanh(x / cap) * cap, in place
/// \note Gemma 4 softcaps its logits at 30.0 on every size. Omitting this does
///       not crash -- it quietly changes the distribution, which is exactly the
///       failure the llama.cpp agreement test exists to catch.
void SoftcapLogits(std::span<float> logits, float cap);

}  // namespace flm::gemma4
