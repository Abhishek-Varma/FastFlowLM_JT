/// \file modeling_gemma3_text_pure_hrx.hpp
/// \brief Gemma3_Text_OnlyPureHrx wrapper: identical behaviour to
///        Gemma3_Text_Only but backed by the gemma_text_npu_pure_hrx engine,
///        whose hot dispatch paths call the native HRX C API directly instead
///        of the flm_rt (hrx::run/runlist) shim.
/// \note  Only load_model (engine instantiation) differs from Gemma3_Text_Only;
///        generation, chat-template and insert logic are inherited unchanged.
#pragma once
#include "AutoModel/modeling_gemma3_text.hpp"
#include "models/gemma_text/gemma_text_npu_pure_hrx.hpp"


/************         gemma3-text (pure/native HRX)          **************/
class Gemma3_Text_OnlyPureHrx : public Gemma3_Text_Only {
public:
    Gemma3_Text_OnlyPureHrx(flm_rt::device* npu_device_inst);

    void load_model(std::string model_path, json model_inf, int default_context_length = -1, bool enable_preemption = false) override;
};
