/// \file modeling_gemma3_pure_hrx.hpp
/// \brief Gemma3PureHrx wrapper: identical behaviour to Gemma3 but backed by the
///        gemma_npu_pure_hrx engine, whose hot decode path calls the native HRX C
///        API directly instead of the flm_rt (hrx::run/runlist) shim.
/// \note  Only create_engine differs from Gemma3; generation, parsing and vision
///        preprocessing are inherited unchanged.
#pragma once
#include "AutoModel/modeling_gemma3.hpp"
#include "models/gemma/gemma_npu_pure_hrx.hpp"


/************            gemma3 (pure/native HRX)            **************/
class Gemma3PureHrx : public Gemma3 {
protected:
    void create_engine() override;

public:
    Gemma3PureHrx(flm_rt::device* npu_device_inst) : Gemma3(npu_device_inst) {}
};
