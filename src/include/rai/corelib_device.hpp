/// \file corelib_device.hpp
/// \brief Access to the NPU device that ryzenai-corelib owns.
/// \note  corelib 0.9's C ABI exposes the hardware context
///        (ryzenai_corelib_get_hardware_context), not an xrt::device. The
///        device itself is ryzenai::corelib::GetDevice(), resolved at link
///        time against the corelib this build links. A null hardware context
///        means there is no NPU; casting that pointer to an xrt::device would
///        hand the engines the wrong object.
#pragma once

#if defined(FLM_USE_HRX)
#error "FLM_ENABLE_RAI requires the XRT backend (FLM_USE_HRX=OFF)"
#endif

#include "device_runtime.hpp"
#include "rai/corelib_runtime.hpp"

namespace ryzenai::corelib {

/// \brief the device corelib initialized; valid until ryzenai_corelib_cleanup()
const xrt::device& GetDevice();

}  // namespace ryzenai::corelib

namespace flm::corelib {

/// \brief the device corelib dispatches on, shared with every engine here
/// \param runtime an initialized corelib runtime
/// \return corelib's device, or nullptr when this machine has no NPU
/// \note Valid until ryzenai_corelib_cleanup(); the process runtime outlives
///       every use of the returned pointer.
/// \note flm_rt is an alias for xrt on this path. corelib hands the device out
///       const and the engines want it mutable; the constness is cast away
///       rather than a second device opened, because a buffer object created
///       against a different xrt::device for the same NPU binds without error
///       and then never completes.
inline flm_rt::device* SharedDevice(const CorelibRuntime& runtime) noexcept {
    if (runtime.api()->functions().get_device() == nullptr) {
        return nullptr;
    }
    return const_cast<flm_rt::device*>(&ryzenai::corelib::GetDevice());
}

}  // namespace flm::corelib
