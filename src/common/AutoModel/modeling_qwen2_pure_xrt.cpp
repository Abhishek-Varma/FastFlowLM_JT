/// \file modeling_qwen2_pure_xrt.cpp
/// \brief Qwen2PureXrt implementation. See modeling_qwen2.cpp for the baseline
///        Qwen2; this wrapper only swaps the engine to qwen2_npu_pure_xrt (native
///        HRX dispatch). Everything else is inherited from Qwen2.

#include "AutoModel/modeling_qwen2_pure_xrt.hpp"


Qwen2PureXrt::Qwen2PureXrt(flm_rt::device* npu_device_inst) : Qwen2(npu_device_inst) {}

void Qwen2PureXrt::load_model(std::string model_path, json model_info, int default_context_length, bool enable_preemption) {
    this->_shared_load_model(model_path, model_info, default_context_length, enable_preemption);

    this->q4nx = std::make_unique<Q4NX>(this->model_path);

    // native/pure XRT engine variant of qwen2_npu
    this->lm_engine = std::make_unique<qwen2_npu_pure_xrt>(*this->lm_config, this->npu.get(), this->MAX_L);

    this->lm_engine->load_weights(*this->q4nx);

    //free the q4nx
    this->q4nx.reset();
    this->lm_engine->clear_context();
    this->setup_tokenizer(model_path);
    this->sampler.reset();

    sampler_config config;
    config.rep_penalty = 1.1;
    config.temperature = 0.6;
    config.top_p = 0.95;
    config.top_k = 10;
    config.rep_penalty_window = 1024;
    config.freq_penalty = 1.1;
    config.freq_penalty_window = 1024;
    this->set_sampler(config);
    for (size_t i = 0; i < PROFILER_TYPE_NUM; i++) {
        this->profiler_list[i].reset();
    }
}
