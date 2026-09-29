#pragma once

#include <cstddef>
#include <cstdint>

namespace flm::gemma4 {

/// The PDI pair Gemma 4's ELFs were built for. Source: corelib python/
/// conftest.py, GEMMA4_PDI. Required by ryzenai_corelib_create_stream since
/// 0.5.0 and given no default by corelib: the silicon ships the same
/// operator under several PDI tags and a model's ELFs live under exactly
/// one. A stream opened on the wrong pair silently loses every shape only
/// that pair ships, surfacing as a missing artifact rather than a wrong
/// answer -- see phi4_rai_constants.hpp's kPrefillPdi/kTokenPdi for the same
/// reasoning applied to Phi-4's own (different) pair.
inline constexpr int kPrefillPdi = 8;
inline constexpr int kTokenPdi = 17;

/// Shared by both shipped rows (E2B and E4B) -- read from the GGUF's own
/// `tokenizer.ggml.tokens` array length in a real load, but the two models
/// this port supports both ship this exact vocabulary.
inline constexpr std::int64_t kVocabularySize = 262144;

/// The group size DynamicDispatch's ELF set at PDI pair (8, 17) ships at.
/// A fact about which kernels exist, not about either model -- see
/// Gemma4Row's doc comment in gemma4_rai_gguf.hpp for how this and the two
/// group fields below it were established.
inline constexpr std::uint32_t kGroupSize = 32;

inline constexpr std::int64_t kMaxSequenceLength = 4096;
inline constexpr std::int64_t kMaxDecodeWindow = 4095;

/// Applied to the final logits on every Gemma 4 size: tanh(x / cap) * cap.
inline constexpr float kLogitSoftcap = 30.0f;

/// rms epsilon, read from `gemma4.attention.layer_norm_rms_epsilon` on both
/// E2B and E4B (9.999999974752427e-07, i.e. 1e-6). NOT Phi-4's 1e-5 --
/// see phi4_rai_constants.hpp's kRmsEpsilon. Kept here only as the value a
/// fixture or a sanity check may compare against; the real loader reads the
/// metadata key rather than trusting this constant.
inline constexpr float kRmsEpsilon = 1.0e-6f;

/// See phi4_rai_constants.hpp for why the intra-packer hint stays at
/// corelib's default of one while the caller runs concurrent creates
/// instead.
inline constexpr std::uint32_t kRequantizeThreads = 0;
inline constexpr std::size_t kWeightCreateConcurrency = 8;

/// \brief bumped whenever the packed bytes change for an unchanged GGUF,
///        corelib version and descriptor; part of the weight-cache layout
/// \note 1: ple's gate and proj are packed from the transpose of the GGUF
///       mapping.
inline constexpr std::uint32_t kPackedLayoutRevision = 1;

}  // namespace flm::gemma4
