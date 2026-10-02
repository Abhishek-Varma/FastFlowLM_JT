/// \file modeling_qwen2vl_pure_hrx.hpp
/// \brief Qwen2VLPureHrx wrapper: identical behaviour to Qwen2VL but backed by
///        the qwen2vl_npu_pure_hrx engine, whose hot decode path calls the
///        native HRX C API directly instead of the flm_rt (hrx::run/runlist)
///        shim.
/// \note  Only create_engine (engine instantiation) differs from Qwen2VL;
///        generation, parsing and vision preprocessing are inherited unchanged.
#pragma once
#include "AutoModel/modeling_qwen2vl.hpp"
#include "models/qwen2vl/qwen2vl_npu_pure_hrx.hpp"


/************            qwen2vl (pure/native HRX)            **************/
class Qwen2VLPureHrx : public Qwen2VL {
protected:
    void create_engine() override;

public:
    Qwen2VLPureHrx(flm_rt::device* npu_device_inst) : Qwen2VL(npu_device_inst) {}
};
