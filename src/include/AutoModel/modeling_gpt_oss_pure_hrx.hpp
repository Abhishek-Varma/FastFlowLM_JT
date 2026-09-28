/// \file modeling_gpt_oss_pure_hrx.hpp
/// \brief GptOssPureHrx wrapper: identical behaviour to GPT_OSS but backed by
///        the gpt_oss_npu_pure_hrx engine (native HRX dispatch instead of the
///        flm_rt hrx::run/runlist shim).
/// \note  Only load_model differs; generation, insert, tool-grammar handling and
///        parsing are inherited. GPT_OSS::insert calls checkpoint()/restore()
///        through the virtual causal_lm engine interface, so it works unchanged
///        for this engine variant.
#pragma once
#include "AutoModel/modeling_gpt_oss.hpp"
#include "models/gpt_oss/gpt_oss_npu_pure_hrx.hpp"


/************              gpt-oss (pure/native HRX)            **************/
class GptOssPureHrx : public GPT_OSS {
public:
    GptOssPureHrx(flm_rt::device* npu_device_inst);

    void load_model(std::string model_path, json model_inf, int default_context_length = -1, bool enable_preemption = false) override;
};
