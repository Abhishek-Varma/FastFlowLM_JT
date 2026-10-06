/// \file modeling_llama3_pure_xrt.cpp
/// \brief Llama3PureXrt implementation. See modeling_llama3.cpp for the baseline
///        Llama3; this wrapper only swaps the engine to llama_npu_pure_xrt (native
///        HRX dispatch). Everything else (generate, insert, chat template,
///        parsing) is inherited from Llama3.

#include "AutoModel/modeling_llama3_pure_xrt.hpp"


Llama3PureXrt::Llama3PureXrt(flm_rt::device* npu_device_inst) : Llama3(npu_device_inst) {}

void Llama3PureXrt::load_model(std::string model_path, json model_info, int default_context_length, bool enable_preemption) {
    this->_shared_load_model(model_path, model_info, default_context_length, enable_preemption);

    this->q4nx = std::make_unique<Q4NX>(this->model_path);
    // native/pure XRT engine variant of llama_npu
    this->lm_engine = std::make_unique<llama_npu_pure_xrt>(*this->lm_config, this->npu.get(), this->MAX_L);

    this->lm_engine->load_weights(*this->q4nx);

    //free the q4nx
    this->q4nx.reset();

    this->lm_engine->clear_context();
    this->setup_tokenizer(model_path);
    this->sampler.reset();

    sampler_config config;
    config.top_k = 40;
    config.top_p = 0.9;
    config.min_p = 0.1;
    config.temperature = 0.8;

    this->set_sampler(config);
    for (size_t i = 0; i < PROFILER_TYPE_NUM; i++) {
        this->profiler_list[i].reset();
    }
}
