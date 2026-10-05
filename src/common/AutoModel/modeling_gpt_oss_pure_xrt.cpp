/// \file modeling_gpt_oss_pure_xrt.cpp
/// \brief GptOssPureXrt implementation. See modeling_gpt_oss.cpp for the baseline
///        GPT_OSS; this wrapper only swaps the engine to gpt_oss_npu_pure_xrt
///        (native XRT per-run dispatch). XRT-backend mirror of
///        modeling_gpt_oss_pure_hrx.cpp.

#include "AutoModel/modeling_gpt_oss_pure_xrt.hpp"


GptOssPureXrt::GptOssPureXrt(flm_rt::device* npu_device_inst) : GPT_OSS(npu_device_inst) {}

void GptOssPureXrt::load_model(std::string model_path, json model_info, int default_context_length, bool enable_preemption) {
    this->model_path = model_path;
    this->_shared_load_model(model_path, model_info, default_context_length, enable_preemption);
    this->q4nx = std::make_unique<Q4NX>(this->model_path);
    // native/pure XRT engine variant of gpt_oss_npu
    this->lm_engine = std::make_unique<gpt_oss_npu_pure_xrt>(*this->lm_config, this->npu.get(), this->MAX_L);
    this->lm_engine->load_weights(*this->q4nx);
    this->q4nx.reset();
    this->tokenizer = std::make_unique<Tokenizer>(model_path);

    this->setup_tokenizer(model_path);
    this->sampler.reset();

    sampler_config config;
    config.top_k = 10;
    config.top_p = 0.95;
    config.min_p = 0.1;
    config.temperature = 0.6;
    config.rep_penalty = 1.05;
    config.freq_penalty = 1.05;

    this->set_sampler(config);
    for (size_t i = 0; i < PROFILER_TYPE_NUM; i++) {
        this->profiler_list[i].reset();
    }
}
