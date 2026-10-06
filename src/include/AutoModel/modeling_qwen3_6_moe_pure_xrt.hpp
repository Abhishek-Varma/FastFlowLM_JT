/// \file modeling_qwen3_6_moe_pure_xrt.hpp
/// \brief Qwen3_6_MOEPureXrt wrapper: identical behaviour to Qwen3_6_MOE but
///        backed by the qwen3_6_moe_npu_pure_xrt engine, whose hot decode path
///        calls the native HRX C API directly instead of the flm_rt
///        (hrx::run/runlist) shim.
/// \note  Only create_engine (engine instantiation) differs from Qwen3_6_MOE;
///        generation, parsing and vision preprocessing are inherited unchanged.
#pragma once
#include "AutoModel/modeling_qwen3_6_moe.hpp"
#include "models/qwen3_6_moe/qwen3_6_moe_npu_pure_xrt.hpp"


/************            qwen3.6-moe (pure/native HRX)            **************/
class Qwen3_6_MOEPureXrt : public Qwen3_6_MOE {
protected:
    void create_engine() override;

public:
    Qwen3_6_MOEPureXrt(flm_rt::device* npu_device_inst) : Qwen3_6_MOE(npu_device_inst) {}
};
