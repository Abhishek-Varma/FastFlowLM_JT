/// \file modeling_hunyuan_pure_xrt.cpp
/// \brief HunyuanPureXrt implementation. See modeling_hunyuan.cpp for the baseline
///        Hunyuan; this wrapper only swaps the engine to hunyuan_npu_pure_xrt
///        (native HRX dispatch).

#include "AutoModel/modeling_hunyuan_pure_xrt.hpp"


HunyuanPureXrt::HunyuanPureXrt(flm_rt::device* npu_device_inst) : Hunyuan(npu_device_inst) {}

void HunyuanPureXrt::load_model(std::string model_path, json model_info, int default_context_length, bool enable_preemption) {
    this->_shared_load_model(model_path, model_info, default_context_length, enable_preemption);

    this->q4nx = std::make_unique<Q4NX>(this->model_path);
    // native/pure XRT engine variant of hunyuan_npu
    this->lm_engine = std::make_unique<hunyuan_npu_pure_xrt>(*this->lm_config, this->npu.get(), this->MAX_L);

    this->lm_engine->load_weights(*this->q4nx);

    // free the mmap'd weights immediately
    this->q4nx.reset();

    this->lm_engine->clear_context();
    this->setup_tokenizer(model_path);
    this->sampler.reset();

    // Defaults published with the checkpoint (general.sampling.* in the GGUF).
    sampler_config config;
    config.top_k = 20;
    config.top_p = 0.6;
    config.temperature = 0.7;
    config.rep_penalty = 1.05;

    this->set_sampler(config);
    for (size_t i = 0; i < PROFILER_TYPE_NUM; i++) {
        this->profiler_list[i].reset();
    }
}
