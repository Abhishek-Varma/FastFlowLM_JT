/// \file modeling_qwen3_pure_hrx.cpp
/// \brief Qwen3PureHrx implementation. See modeling_qwen3.cpp for the baseline
///        Qwen3; this wrapper only swaps the engine to qwen3_npu_pure_hrx (native
///        HRX dispatch) and casts to it for checkpoint/restore. Everything else
///        (generate, chat template, parsing) is inherited from Qwen3.

#include "AutoModel/modeling_qwen3_pure_hrx.hpp"


Qwen3PureHrx::Qwen3PureHrx(flm_rt::device* npu_device_inst) : Qwen3(npu_device_inst) {}

void Qwen3PureHrx::load_model(std::string model_path, json model_info, int default_context_length, bool enable_preemption) {
    this->_shared_load_model(model_path, model_info, default_context_length, enable_preemption);

    this->q4nx = std::make_unique<Q4NX>(this->model_path);
    // native/pure HRX engine variant of qwen3_npu
    this->lm_engine = std::make_unique<qwen3_npu_pure_hrx>(*this->lm_config, this->npu.get(), this->MAX_L);

    this->lm_engine->load_weights(*this->q4nx);

    //free the q4nx
    this->q4nx.reset();
    this->lm_engine->clear_context();
    this->setup_tokenizer(model_path);
    this->sampler.reset();

    this->enable_think = (model_info["size"] == 600000000) ? false : true;
    this->enable_tool = (model_info["size"] > 1700000000) ? true : false;

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

bool Qwen3PureHrx::insert(chat_meta_info_t& meta_info, lm_uniform_input_t& input, std::function<bool()> is_cancelled) {
    // preprocess
    this->profiler_list[TKOEN_ENCODE_TIME].start();
    std::string templated_text;
    if (input.messages.empty() && input.prompt.empty()) {
        header_print("WARNING", "No messages or prompt provided");
        return false;
    }
    if (!input.messages.empty()) { // already a formated messages, usually from REST API
        templated_text = this->apply_chat_template(input.messages, input.tools);
    }
    else if (!input.prompt.empty()) { // a pure text, usually from the cli
        nlohmann::ordered_json messages;

        messages.push_back({ {"role", "user"}, {"content", input.prompt} });
        templated_text = this->apply_chat_template(messages);
    }

    std::vector<int> tokens = this->tokenizer->encode(templated_text);

    this->profiler_list[TKOEN_ENCODE_TIME].stop(tokens.size());

    // hardware
    int restore_idx = -1;
    qwen3_npu_pure_hrx *qwen3_engine = dynamic_cast<qwen3_npu_pure_hrx*>(this->lm_engine.get());

    if (meta_info.restore_allowed) {
        restore_idx = qwen3_engine->restore();
        this->total_tokens = restore_idx;
        this->token_history = checkpoint_his; // restore the token history to be consistent with the restored KV cache, which is crucial for correct functioning of _shared_insert's prefix-matching logic
    }

    size_t n = tokens.size();
    tokens.resize(n - (this->enable_think ? 0 : 4));

    bool success = this->_shared_insert(meta_info, tokens, is_cancelled, nullptr);

    checkpoint_his = token_history;
    int checkpoint_idx = qwen3_engine->checkpoint();
    return success;
}
