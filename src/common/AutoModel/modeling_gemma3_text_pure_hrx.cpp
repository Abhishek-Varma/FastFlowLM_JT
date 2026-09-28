/// \file modeling_gemma3_text_pure_hrx.cpp
/// \brief Gemma3_Text_OnlyPureHrx implementation. See modeling_gemma3_text.cpp
///        for the baseline Gemma3_Text_Only; this wrapper only swaps the engine
///        to gemma_text_npu_pure_hrx (native HRX dispatch). Everything else
///        (generate, chat template, insert) is inherited from Gemma3_Text_Only.

#include "AutoModel/modeling_gemma3_text_pure_hrx.hpp"


Gemma3_Text_OnlyPureHrx::Gemma3_Text_OnlyPureHrx(flm_rt::device* npu_device_inst) : Gemma3_Text_Only(npu_device_inst) {}

void Gemma3_Text_OnlyPureHrx::load_model(std::string model_path, json model_info, int default_context_length, bool enable_preemption) {

    this->_shared_load_model(model_path, model_info, default_context_length, enable_preemption);

    this->q4nx = std::make_unique<Q4NX>(this->model_path);
    // native/pure HRX engine variant of gemma_text_npu
    this->lm_engine = std::make_unique<gemma_text_npu_pure_hrx>(*this->lm_config, this->npu.get(), this->MAX_L);

    this->lm_engine->load_weights(*this->q4nx);

    //free the q4nx
    this->q4nx.reset();
    this->lm_engine->clear_context();
    this->setup_tokenizer(model_path);
    this->sampler.reset();

    sampler_config config;
    config.top_k = 40;
    config.top_p = 0.95;
    config.min_p = 0.1;
    config.temperature = 0.8;
    config.rep_penalty = 1.05;
    config.freq_penalty = 1.05;

    this->set_sampler(config);
    for (size_t i = 0; i < PROFILER_TYPE_NUM; i++) {
        this->profiler_list[i].reset();
    }
}
