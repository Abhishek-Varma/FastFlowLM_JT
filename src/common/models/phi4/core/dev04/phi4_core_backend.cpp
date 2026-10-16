/// \file phi4_core_backend.cpp
/// \brief Phi-4-mini on corelib's "phi4mini" model
#include "models/phi4/core/dev04/phi4_core_backend.hpp"

#include "core/corelib_model_backend.hpp"

namespace flm::phi4 {

/// \note corelib reads the catalog's GGUF, Q8_0 or Q4_G128, and checks its
///       shapes and metadata itself. A multi-row pass starts only at position
///       0, so a later prompt re-runs the conversation with it, as the per-op
///       engine did.
flm::backend::BackendFactory core_factory() {
    return core::Factory({
        .model = {"phi4mini", "Phi-4", kCoreContextLimit, core::Continuation::kFromStart,
                  core::Rewind::kSeek},
        .context_rule = "1..4096",
        // <|end|>, <|endoftext|>
        .eos_ids = {200020, 199999},
    });
}

}  // namespace flm::phi4
