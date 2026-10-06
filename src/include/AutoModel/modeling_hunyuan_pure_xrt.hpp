/// \file modeling_hunyuan_pure_xrt.hpp
/// \brief HunyuanPureXrt wrapper: identical behaviour to Hunyuan but backed by
///        the hunyuan_npu_pure_xrt engine (native HRX dispatch instead of the
///        flm_rt hrx::run/runlist shim).
/// \note  Only load_model differs; generation, insert, prefix-pinning and parsing
///        are inherited. Hunyuan::insert / _pin_system_prefix call checkpoint()/
///        restore() through the virtual causal_lm engine interface, so they work
///        unchanged for this engine variant.
#pragma once
#include "AutoModel/modeling_hunyuan.hpp"
#include "models/hunyuan/hunyuan_npu_pure_xrt.hpp"


/************              hunyuan-dense (pure/native HRX)            **************/
class HunyuanPureXrt : public Hunyuan {
public:
    HunyuanPureXrt(flm_rt::device* npu_device_inst);

    void load_model(std::string model_path, json model_inf, int default_context_length = -1, bool enable_preemption = false) override;
};
