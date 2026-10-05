/// \file modeling_gpt_oss_pure_xrt.hpp
/// \brief GptOssPureXrt wrapper: identical behaviour to GPT_OSS but backed by
///        the gpt_oss_npu_pure_xrt engine (native XRT per-run dispatch instead
///        of the flm_rt xrt::runlist batched layer). XRT-backend mirror of
///        GptOssPureHrx.
/// \note  Only load_model differs; generation, insert, tool-grammar handling and
///        parsing are inherited. GPT_OSS::insert calls checkpoint()/restore()
///        through the virtual causal_lm engine interface, so it works unchanged
///        for this engine variant.
#pragma once
#include "AutoModel/modeling_gpt_oss.hpp"
#include "models/gpt_oss/gpt_oss_npu_pure_xrt.hpp"


/************              gpt-oss (pure/native XRT)            **************/
class GptOssPureXrt : public GPT_OSS {
public:
    GptOssPureXrt(flm_rt::device* npu_device_inst);

    void load_model(std::string model_path, json model_inf, int default_context_length = -1, bool enable_preemption = false) override;
};
