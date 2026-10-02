/// \file modeling_qwen3_5vl_pure_hrx.hpp
/// \brief Qwen3_5VLPureHrx wrapper: identical behaviour to Qwen3_5VL but backed
///        by the qwen3_5vl_npu_pure_hrx engine, whose hot decode path calls the
///        native HRX C API directly instead of the flm_rt (hrx::run/runlist)
///        shim.
/// \note  Only create_engine (engine instantiation) differs from Qwen3_5VL;
///        generation, parsing and vision preprocessing are inherited unchanged.
#pragma once
#include "AutoModel/modeling_qwen3_5vl.hpp"
#include "models/qwen3_5vl/qwen3_5vl_npu_pure_hrx.hpp"


/************            qwen3.5-vl (pure/native HRX)            **************/
class Qwen3_5VLPureHrx : public Qwen3_5VL {
protected:
    void create_engine() override;

public:
    Qwen3_5VLPureHrx(flm_rt::device* npu_device_inst) : Qwen3_5VL(npu_device_inst) {}
};
