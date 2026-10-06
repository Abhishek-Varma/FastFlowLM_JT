/// \file modeling_nanbeige_pure_xrt.cpp
/// \brief NanbeigePureXrt implementation. See modeling_nanbeige.cpp for the
///        baseline Nanbeige; this wrapper only swaps the engine to
///        nanbeige_npu_pure_xrt (native HRX dispatch).

#include "AutoModel/modeling_nanbeige_pure_xrt.hpp"


NanbeigePureXrt::NanbeigePureXrt(flm_rt::device* npu_device_inst) : Nanbeige(npu_device_inst) {}

void NanbeigePureXrt::load_model(std::string model_path, json model_info, int default_context_length, bool enable_preemption) {
    this->_shared_load_model(model_path, model_info, default_context_length, enable_preemption);

    this->q4nx = std::make_unique<Q4NX>(this->model_path);
    // native/pure XRT engine variant of nanbeige_npu
    this->lm_engine = std::make_unique<nanbeige_npu_pure_xrt>(*this->lm_config, this->npu.get(), this->MAX_L);

    this->lm_engine->load_weights(*this->q4nx);

    //free the q4nx
    this->q4nx.reset();

    this->lm_engine->clear_context();
    this->setup_tokenizer(model_path);
    this->sampler.reset();

    sampler_config config;
    config.top_k = 20;
    config.top_p = 0.95;
    config.min_p = 0.0;
    config.temperature = 0.6;

    this->set_sampler(config);
    for (size_t i = 0; i < PROFILER_TYPE_NUM; i++) {
        this->profiler_list[i].reset();
    }
}
