/// \file modeling_gemma4_12b_pure_hrx.hpp
/// \brief Gemma4_12BPureHrx wrapper: identical behaviour to Gemma4_12B but backed by
///        the gemma4_12b_npu_pure_hrx engine, whose hot decode path calls the native
///        HRX C API directly instead of the flm_rt (hrx::run/runlist) shim.
/// \note  Only create_engine differs from Gemma4_12B; generation, parsing and
///        image/audio preprocessing are inherited unchanged.
#pragma once
#include "AutoModel/modeling_gemma4_12b.hpp"
#include "models/gemma4_12b/gemma4_12b_npu_pure_hrx.hpp"


/************            gemma4_12b (pure/native HRX)            **************/
class Gemma4_12BPureHrx : public Gemma4_12B {
protected:
    void create_engine() override;

public:
    Gemma4_12BPureHrx(flm_rt::device* npu_device_inst) : Gemma4_12B(npu_device_inst) {}
};
