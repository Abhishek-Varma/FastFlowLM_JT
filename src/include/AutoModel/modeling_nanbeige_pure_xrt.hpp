/// \file modeling_nanbeige_pure_xrt.hpp
/// \brief NanbeigePureXrt wrapper: identical behaviour to Nanbeige but backed by
///        the nanbeige_npu_pure_xrt engine (native HRX dispatch instead of the
///        flm_rt hrx::run/runlist shim).
/// \note  Only load_model differs; generation, insert and parsing are inherited.
///        The base Nanbeige::insert calls checkpoint()/restore() through the
///        virtual causal_lm engine interface, so it works unchanged here.
#pragma once
#include "AutoModel/modeling_nanbeige.hpp"
#include "models/nanbeige/nanbeige_npu_pure_xrt.hpp"


/************              nanbeige (pure/native HRX)            **************/
class NanbeigePureXrt : public Nanbeige {
public:
    NanbeigePureXrt(flm_rt::device* npu_device_inst);

    void load_model(std::string model_path, json model_inf, int default_context_length = -1, bool enable_preemption = false) override;
};
