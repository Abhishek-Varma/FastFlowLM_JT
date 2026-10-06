/// \file modeling_qwen3vl_pure_xrt.hpp
/// \brief Qwen3VLPureXrt wrapper: identical behaviour to Qwen3VL but backed by
///        the qwen3vl_npu_pure_xrt engine, whose hot decode path calls the
///        native HRX C API directly instead of the flm_rt (hrx::run/runlist)
///        shim.
/// \note  Only create_engine (engine instantiation) differs from Qwen3VL;
///        generation, parsing and vision preprocessing are inherited unchanged.
#pragma once
#include "AutoModel/modeling_qwen3vl.hpp"
#include "models/qwen3vl/qwen3vl_npu_pure_xrt.hpp"


/************            qwen3vl (pure/native HRX)            **************/
class Qwen3VLPureXrt : public Qwen3VL {
protected:
    void create_engine() override;

public:
    Qwen3VLPureXrt(flm_rt::device* npu_device_inst) : Qwen3VL(npu_device_inst) {}
};
