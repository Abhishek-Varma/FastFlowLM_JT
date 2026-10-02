/// \file gemma4_rai_backend.hpp
/// \brief The ryzenai-corelib backend for Gemma 4 (E2B and E4B)
/// \note Compiled only when FLM_ENABLE_RAI is on. A ModelBackend owns one
///       engine AND every rule for driving it, so everything that would
///       otherwise have to become a rai special case inside the gemma4e
///       frontend -- the 4095-token decode window, the no-preemption rule, the
///       package-verified EOS ids, the poisoning state -- is stated here
///       instead. The frontend stays backend-agnostic: there is no
///       `#if FLM_ENABLE_RAI` anywhere under src/common/AutoModel/ except
///       builtin_backends.cpp, which is the one file that knows both a family
///       name and an engine type.
#pragma once

#include "AutoModel/model_backend.hpp"
#include "models/gemma4/rai/aie_next/gemma4_rai_constants.hpp"

#include <cstdint>

namespace flm::gemma4 {

/// \brief the largest context this backend can hold
/// \note THE ENGINE'S OWN CONSTANT, not a second copy of it. `gemma4_rai`'s
///       Impl refuses a `max_length` above `kMaxSequenceLength` and allocates
///       every KV cache and rotary table at exactly that extent, so a traits
///       ceiling written independently could drift above what the engine will
///       accept -- and the frontend checks the TRAITS, before the factory
///       runs, so the drift would surface as a load that passes the frontend's
///       range check and then throws out of the engine constructor.
inline constexpr std::uint32_t kRaiContextLimit =
    static_cast<std::uint32_t>(kMaxSequenceLength);

/// \brief the largest number of tokens corelib will decode
/// \note Likewise `kMaxDecodeWindow`, one below the context: the last position
///       in the window has to be readable, so a decode that filled it would
///       have nowhere to put the next token.
inline constexpr std::uint32_t kRaiDecodeLimit =
    static_cast<std::uint32_t>(kMaxDecodeWindow);

/// \brief what the frontend must know before building this backend
/// \return no xclbin, no preemption, 4096-token context ceiling
/// \note Pure data, and inline on purpose: `AutoModel::_shared_load_backend`
///       reads it while assembling the BackendContext -- that is, BEFORE any
///       engine exists -- so it cannot be a virtual on ModelBackend. Being in
///       the header also lets a test read it without linking the engine.
inline flm::backend::BackendTraits rai_traits() {
    flm::backend::BackendTraits traits;
    traits.needs_npu_xclbin = false;      // no xclbin manager is built
    traits.supports_preemption = false;   // rejected before the factory runs
    traits.max_context_length = kRaiContextLimit;  // likewise
    return traits;
}

/// \brief a factory building the rai backend
/// \return a factory suitable for BackendRegistry::register_backend
flm::backend::BackendFactory rai_factory();

}  // namespace flm::gemma4
