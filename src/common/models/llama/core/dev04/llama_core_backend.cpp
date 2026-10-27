/// \file llama_core_backend.cpp
/// \brief Llama 3.2 1B and 3B, and Llama 3.1 8B, on corelib's "llama" model
#include "models/llama/core/dev04/llama_core_backend.hpp"

#include "core/corelib_model_backend.hpp"

namespace flm::llama {

/// \note A multi-row pass starts only at position 0, so a later prompt re-runs
///       the conversation with it.
flm::backend::BackendFactory core_factory() {
    return core::Factory({
        .model = {"llama", "Llama", kCoreContextLimit, core::Continuation::kFromStart,
                  core::Rewind::kSeek},
        .context_rule = "1..4096",
        // <|eot_id|>, <|eom_id|>, <|end_of_text|>: eos_token_id in the
        // catalog's config.json
        .eos_ids = {128009, 128008, 128001},
    });
}

}  // namespace flm::llama
