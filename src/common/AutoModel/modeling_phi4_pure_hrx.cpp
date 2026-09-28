/// \file modeling_phi4_pure_hrx.cpp
/// \brief Phi4PureHrx implementation. See modeling_phi4.cpp for the baseline Phi4;
///        this wrapper only swaps the engine to phi4_npu_pure_hrx (native HRX
///        dispatch). Everything else is inherited from Phi4.

#include "AutoModel/modeling_phi4_pure_hrx.hpp"


Phi4PureHrx::Phi4PureHrx(flm_rt::device* npu_device_inst) : Phi4(npu_device_inst) {}

void Phi4PureHrx::load_model(std::string model_path, json model_info, int default_context_length, bool enable_preemption) {
    this->_shared_load_model(model_path, model_info, default_context_length, enable_preemption);

    this->q4nx = std::make_unique<Q4NX>(this->model_path);
    // native/pure HRX engine variant of phi4_npu
    this->lm_engine = std::make_unique<phi4_npu_pure_hrx>(*this->lm_config, this->npu.get(), this->MAX_L);
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
