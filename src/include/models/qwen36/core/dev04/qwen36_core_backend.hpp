/// \file qwen36_core_backend.hpp
/// \brief The ryzenai-corelib backend for Qwen3.6-35B-A3B text
/// \note Compiled only when FLM_ENABLE_CORE is on.
#pragma once

#include "AutoModel/model_backend.hpp"

#include <cstdint>

namespace flm::qwen36 {

/// \note corelib takes up to the model's own 262144; this stays at 131072,
///       the most FastFlowLM has run on the NPU.
inline constexpr std::uint32_t kCoreContextLimit = 131072;

inline flm::backend::BackendTraits core_traits() {
    flm::backend::BackendTraits traits;
    traits.needs_npu_xclbin = false;
    traits.supports_preemption = false;
    traits.max_context_length = kCoreContextLimit;
    return traits;
}

flm::backend::BackendFactory core_factory();

}  // namespace flm::qwen36
