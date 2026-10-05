/// \file corelib_device.hpp
/// \brief Access to the NPU device that ryzenai-corelib owns.
/// \note  The C ABI exposes the hardware context
///        (ryzenai_corelib_get_hardware_context), which is corelib's
///        xrt::hw_context. The device is the shared handle that context was
///        opened on. A null hardware context means there is no NPU; casting
///        that pointer to an xrt::device would hand the engines the wrong
///        object.
#pragma once

#if defined(FLM_USE_HRX)
#error "FLM_ENABLE_RAI requires the XRT backend (FLM_USE_HRX=OFF)"
#endif

#include "device_runtime.hpp"
#include "rai/corelib_runtime.hpp"

#include "xrt/xrt_hw_context.h"

namespace flm::corelib {

/// \brief the device corelib dispatches on, shared with every engine here
/// \param runtime an initialized corelib runtime
/// \return corelib's device, or nullptr when this machine has no NPU
/// \note The returned pointer stays valid for the process. hw_context::get_device
///       returns the handle by value, so a copy is kept here; it is the same
///       NPU corelib dispatches on, not a second open.
/// \note flm_rt is an alias for xrt on this path. The engines want the device
///       mutable. A buffer object created against a different xrt::device for
///       the same NPU binds without error and then never completes.
inline flm_rt::device* SharedDevice(const CorelibRuntime& runtime) {
    const void* hardware = runtime.api()->functions().get_device();
    if (hardware == nullptr) {
        return nullptr;
    }
    static flm_rt::device device =
        static_cast<const xrt::hw_context*>(hardware)->get_device();
    return &device;
}

}  // namespace flm::corelib
