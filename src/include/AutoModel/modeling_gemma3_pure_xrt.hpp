/// \file modeling_gemma3_pure_xrt.hpp
/// \brief Gemma3PureXrt wrapper: identical behaviour to Gemma3 but backed by the
///        gemma_npu_pure_xrt engine, whose hot decode path dispatches the native
///        xrt::run objects directly instead of the flm_rt (xrt::runlist) shim.
///        XRT-backend mirror of Gemma3PureHrx.
/// \note  Only create_engine differs from Gemma3; generation, parsing and vision
///        preprocessing are inherited unchanged.
#pragma once
#include "AutoModel/modeling_gemma3.hpp"
#include "models/gemma/gemma_npu_pure_xrt.hpp"


/************            gemma3 (pure/native XRT)            **************/
class Gemma3PureXrt : public Gemma3 {
protected:
    void create_engine() override;

public:
    Gemma3PureXrt(flm_rt::device* npu_device_inst) : Gemma3(npu_device_inst) {}
};
