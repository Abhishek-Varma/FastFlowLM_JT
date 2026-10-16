/// \file qwen3_core_backend.cpp
/// \brief Qwen3 0.6B, 1.7B, 4B and 8B on corelib's "qwen3" model
#include "models/qwen3/core/dev04/qwen3_core_backend.hpp"

#include "core/corelib_model_backend.hpp"

namespace flm::qwen3 {

/// \note A multi-row pass starts only at position 0, so a later prompt re-runs
///       the conversation with it, as the per-op engine did.
flm::backend::BackendFactory core_factory() {
    return core::Factory({
        .model = {"qwen3", "Qwen3", kCoreContextLimit, core::Continuation::kFromStart,
                  core::Rewind::kSeek},
        .context_rule = "1..4096",
        // <|im_end|>, <|endoftext|>
        .eos_ids = {151645, 151643},
    });
}

}  // namespace flm::qwen3
