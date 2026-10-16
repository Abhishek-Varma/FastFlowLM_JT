/// \file qwen35_core_backend.cpp
/// \brief Dense Qwen3.5 on corelib's "qwen35" model
#include "models/qwen35/core/dev04/qwen35_core_backend.hpp"

#include "core/corelib_model_backend.hpp"

#include <string>

namespace flm::qwen35 {

/// \note The linear-attention state can neither rewind nor skip and corelib
///       keeps it outside any state slot, so going back means replaying.
flm::backend::BackendFactory core_factory() {
    return core::Factory({
        .model = {"qwen35", "Qwen3.5", kCoreContextLimit, core::Continuation::kAnywhere,
                  core::Rewind::kReplay},
        .context_rule = "1.." + std::to_string(kCoreContextLimit),
        .eos_ids = {248046, 248044},
    });
}

}  // namespace flm::qwen35
