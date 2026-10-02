/// \file modeling_gemma4e_pure_hrx.hpp
/// \brief Gemma4ePureHrx wrapper: identical behaviour to Gemma4e but backed by the
///        gemma4e_npu_pure_hrx engine, whose hot decode path calls the native HRX C
///        API directly instead of the flm_rt (hrx::run/runlist) shim.
/// \note  Only create_engine differs from Gemma4e; generation, parsing and
///        image/audio preprocessing are inherited unchanged.
#pragma once
#include "AutoModel/modeling_gemma4e.hpp"
#include "models/gemma4e/gemma4e_npu_pure_hrx.hpp"


/************            gemma4e (pure/native HRX)            **************/
class Gemma4ePureHrx : public Gemma4e {
protected:
    void create_engine() override;

public:
    Gemma4ePureHrx(flm_rt::device* npu_device_inst) : Gemma4e(npu_device_inst) {}
};
