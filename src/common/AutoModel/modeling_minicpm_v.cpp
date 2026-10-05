/// \file modeling_minicpm_v.cpp
/// \brief MiniCPM_V class
/// \author FastFlowLM Team
/// \date 2026-10-02
/// \version 0.9.28
/// \note This is a source file for the MiniCPM_V class


#include "AutoModel/modeling_minicpm_v.hpp"

#include <cstdlib>
#include <fstream>


/************              MiniCPM_V family            **************/
MiniCPM_V::MiniCPM_V(flm_rt::device* npu_device_inst) : AutoModel(npu_device_inst, "MiniCPM_V") {}

void MiniCPM_V::load_model(std::string model_path, json model_info, int default_context_length, bool enable_preemption, const std::string& backend) {
    this->_shared_load_backend(model_path, model_info, default_context_length, enable_preemption, backend);
    this->setup_tokenizer(model_path);
    this->sampler.reset();

    // The shipped template renders tools in the same XML format as Qwen3.5.
    this->enable_tool = true;

    // MiniCPM-V ships no sampling defaults of its own; its text tower is
    // Qwen3.5-0.8B's, so start from the Qwen3.5 non-thinking recommendation.
    sampler_config config;
    config.top_k = 20;
    config.top_p = 0.8;
    config.min_p = 0.0;
    config.temperature = 0.7;
    config.rep_penalty = 1.0;
    config.freq_penalty = 1.0;
    config.pre_penalty = 1.5f;

    this->set_sampler(config);
    for (size_t i = 0; i < PROFILER_TYPE_NUM; i++) {
        this->profiler_list[i].reset();
    }
}

void MiniCPM_V::setup_tokenizer(std::string model_path) {
    auto tokenizer_config = this->_shared_setup_tokenizer(model_path);
}

std::string MiniCPM_V::apply_chat_template(nlohmann::ordered_json& messages, nlohmann::ordered_json tools) {
    minja::chat_template_inputs inputs;
    inputs.add_generation_prompt = true;
    inputs.messages = messages;
    inputs.extra_context = this->extra_context;
    inputs.extra_context["enable_thinking"] = this->enable_think;
    if (!tools.empty() && this->enable_tool)
        inputs.tools = tools;
    return this->chat_tmpl->apply(inputs);
}

void MiniCPM_V::fail_inference() {
    const bool poisoned = this->backend_ && this->backend_->poisoned();
    this->_shared_after_inference_failure(poisoned);
    throw ModelRequestError(500, true, poisoned
        ? "Inference failed; unload/reload is required because the model is poisoned"
        : "Inference failed; the current conversation was cleared");
}

bool MiniCPM_V::insert(chat_meta_info_t& meta_info, lm_uniform_input_t& input, std::function<bool()> is_cancelled) {
    this->_shared_guard_poisoned();
    constexpr int image_pad = 248056;
    // preprocess
    this->profiler_list[TKOEN_ENCODE_TIME].start();
    std::string templated_text;
    if (input.messages.empty() && input.prompt.empty()) {
        header_print("WARNING", "No messages or prompt provided");
        return false;
    }

    if (!input.audios.empty()) {
        header_print("WARNING", "MiniCPM-V has no audio input, ignoring " << input.audios.size() << " audio input(s)");
        input.audios.clear();
        input.audio_payload_types.clear();
    }

    // is_vlm is set from config.json's vision_model_weight; without it the engine
    // has no vision tower, so images are dropped here rather than failing there.
    const bool vlm = this->lm_config && this->lm_config->get<bool>("is_vlm", false);

    // One layout per picture, in prompt order; the payload holds their VIEWS.
    minicpm_v_image_payload_t image_payload;
    image_payload.num_images = 0;
    std::vector<image_layout> layouts;
    auto add_image = [&](const std::string& source, bool base64) {
        image_data_t chw;
        if (!this->load_image(source, base64, chw)) {
            header_print("ERROR", "Skipping image that failed to load");
            return false;
        }
        layouts.push_back(this->preprocess_image(chw, image_payload));
        image_reader_.recycle(chw);
        return true;
    };

    if (!input.messages.empty()) { // already a formated messages, usually from REST API
        nlohmann::ordered_json messages = nlohmann::ordered_json::array();
        for (auto& message : input.messages) {
            message.erase("audios");
            if (!message.contains("images")) {
                messages.push_back(message);
                continue;
            }
            if (!vlm) {
                header_print("WARNING", "this MiniCPM-V package has no vision tower, ignoring " << message["images"].size() << " image(s)");
                message.erase("images");
                messages.push_back(message);
                continue;
            }
            // Decode/preprocess up front so an invalid image never gets a
            // placeholder in the template below.
            nlohmann::ordered_json content = nlohmann::ordered_json::array();
            for (const auto& img : message["images"]) {
                if (add_image(img.get<std::string>(), true)) {
                    content.push_back({ {"type", "image"} });
                }
            }
            content.push_back({ {"type", "text"}, {"text", message["content"]} });
            messages.push_back({ {"role", message["role"]}, {"content", content} });
        }
        templated_text = this->apply_chat_template(messages, input.tools);
    }
    else if (!input.prompt.empty()) { // a pure text, usually from the cli
        nlohmann::ordered_json messages;
        nlohmann::ordered_json content = nlohmann::ordered_json::array();
        if (!input.images.empty() && !vlm) {
            header_print("WARNING", "this MiniCPM-V package has no vision tower, ignoring " << input.images.size() << " image(s)");
        }
        else {
            for (const auto& path : input.images) {
                if (add_image(path, false)) {
                    content.push_back({ {"type", "image"} });
                }
            }
        }
        content.push_back({ {"type", "text"}, {"text", input.prompt} });
        messages.push_back({ {"role", "user"}, {"content", content} });
        templated_text = this->apply_chat_template(messages);
    }
    input.images.clear();
    input.image_payload_types.clear();

    // Expand each picture's single <|image_pad|> to the processor's placeholder.
    std::vector<int> tokens_init = this->tokenizer->encode(templated_text);
    std::vector<int> tokens;
    {
        size_t extra = 0;
        for (const auto& l : layouts) extra += (size_t)l.tokens + 16 + 3 * (size_t)l.views;
        tokens.reserve(tokens_init.size() + extra);
        size_t picture = 0;
        for (int id : tokens_init) {
            if (id != image_pad) {
                tokens.push_back(id);
                continue;
            }
            if (picture >= layouts.size()) {
                header_print("ERROR", "the prompt has more <|image_pad|> than images");
                return false;
            }
            const std::vector<int> ph = this->image_placeholder(layouts[picture], (int)picture);
            tokens.insert(tokens.end(), ph.begin(), ph.end());
            picture++;
        }
        if (picture != layouts.size()) {
            header_print("ERROR", "the prompt has " << picture << " <|image_pad|> for " << layouts.size() << " images");
            return false;
        }
    }

    this->profiler_list[TKOEN_ENCODE_TIME].stop(tokens.size());

    if (meta_info.restore_allowed) {
        const int restore_idx = this->lm_engine->restore();
        if (restore_idx >= 0) {
            this->total_tokens = restore_idx;
            this->token_history = checkpoint_his; // restore the token history to be consistent with the restored KV cache, which is crucial for correct functioning of _shared_insert's prefix-matching logic
        }
    }

    // The generation prompt ends in "<think>\n" (2 tokens) or, with thinking
    // off, "<think>\n\n</think>\n\n" (4 tokens). Leave them out of the prompt so
    // the checkpoint lands right after "<|im_start|>assistant\n"; decode()
    // feeds them back.
    tokens.resize(tokens.size() - (this->enable_think ? 2 : 4));

    // ----------------------------------------------------------------------
    // Prompt-cache aware image alignment. _shared_insert erases the prefix of
    // `tokens` that matches token_history -- only when it matches ALL of
    // token_history, else it clears the context and skips nothing. The payload
    // must lose exactly the pictures that erased prefix fully covers (all of
    // their views) so it lines up with the image rows that are prefilled.
    //
    // This is _shared_insert's rule on _shared_insert's inputs: token_history
    // AFTER the restore above and `tokens` AFTER the think trim. Computing it
    // against checkpoint_his instead (Qwen3_5VL's way) disagrees in the CLI,
    // where token_history also holds the previous answer: a repeated question
    // then "hits" here, its image is dropped, _shared_insert clears and
    // re-prefills the whole prompt, and the engine is handed image rows with
    // no pixels.
    // ----------------------------------------------------------------------
    size_t prefix_skip_count = 0;
    {
        const size_t idx = this->token_history.size();
        while (prefix_skip_count < idx && prefix_skip_count < tokens.size() &&
               tokens[prefix_skip_count] == this->token_history[prefix_skip_count]) {
            prefix_skip_count++;
        }
        if (prefix_skip_count != idx) prefix_skip_count = 0;

        if (prefix_skip_count > 0 && !layouts.empty()) {
            int skipped_rows = 0;
            for (size_t i = 0; i < prefix_skip_count; i++) skipped_rows += (tokens[i] == image_pad);
            size_t drop_pictures = 0, drop_views = 0, drop_values = 0;
            int consumed = 0;
            for (const auto& l : layouts) {
                if (consumed + l.tokens > skipped_rows) break;
                consumed += l.tokens;
                drop_pictures++;
                drop_views += (size_t)l.views;
                drop_values += l.values;
            }
            if (drop_pictures > 0) {
                image_payload.images.erase(image_payload.images.begin(), image_payload.images.begin() + drop_views);
                image_payload._data__processed.erase(image_payload._data__processed.begin(),
                                                     image_payload._data__processed.begin() + drop_values);
                image_payload.num_images = (unsigned)image_payload.images.size();
                header_print("FLM", "Prompt-cache hit: dropped " << drop_pictures << " cached image(s) from payload");
            }
        }
    }

    // The payload rides on the first prefill chunk only, so that chunk must hold
    // every image GROUP whole -- through the </slice> / </image> that close it,
    // which the canvas m-rope positions too -- not just up to the last pad row.
    int first_len_run = 0;
    {
        int last = -1;
        for (int i = (int)prefix_skip_count; i < (int)tokens.size(); i++) {
            if (tokens[i] == image_pad) last = i;
        }
        if (last >= 0) {
            auto structural = [](int t) {
                return t == 248078 || t == 248079 || t == 248088 || t == 248089;
            };
            while (last + 1 < (int)tokens.size() && structural(tokens[last + 1])) last++;
            first_len_run = last + 1 - (int)prefix_skip_count;
        }
    }
    const bool has_images = !image_payload.images.empty();

    // FLM_MINICPM_V_DUMP=<dir>: the payload and the prompt exactly as the engine
    // gets them, for scoring the preprocessing against the HF processor
    // (pixels.bf16 raw, views.txt "h w" per view, tokens.txt one id per line).
    if (const char* dump = std::getenv("FLM_MINICPM_V_DUMP"); dump && *dump) {
        const std::string dir(dump);
        std::ofstream(dir + "/pixels.bf16", std::ios::binary)
            .write(reinterpret_cast<const char*>(image_payload._data__processed.data()),
                   image_payload._data__processed.size() * sizeof(bf16));
        std::ofstream vf(dir + "/views.txt");
        for (const auto& v : image_payload.images) vf << v.grid_h << " " << v.grid_w << "\n";
        std::ofstream tf(dir + "/tokens.txt");
        for (int t : tokens) tf << t << "\n";
        header_print("FLM", "dumped the image payload and prompt to " << dir);
    }

    bool success = false;
    try {
        success = this->_shared_insert(meta_info, tokens, is_cancelled, has_images ? &image_payload : nullptr,
                                       has_images ? first_len_run : 0, input.requested_max_new_tokens);
    } catch (const ModelRequestError&) {
        throw;
    } catch (...) {
        this->fail_inference();
    }

    checkpoint_his = token_history;
    this->lm_engine->checkpoint();
    return success;
}

std::string MiniCPM_V::generate(chat_meta_info_t& meta_info, int length_limit, std::ostream& os, std::function<bool()> is_cancelled) {
    this->_shared_guard_poisoned();
    try {
        return this->decode(meta_info, length_limit, os, std::move(is_cancelled));
    } catch (const ModelRequestError&) {
        throw;
    } catch (...) {
        this->fail_inference();
    }
}

std::string MiniCPM_V::decode(chat_meta_info_t& meta_info, int length_limit, std::ostream& os, std::function<bool()> is_cancelled) {
    std::string result;
    assert(this->last_token != -1);
    stop_reason_t reason = EOT_DETECTED;

    this->profiler_list[DECODING_TIME].reset();
    this->profiler_list[TKOEN_DECODE_TIME].reset();
    // Some backends refuse to decode past a limit of their own, below MAX_L.
    const uint32_t decode_cap = this->decode_cap();

    // The tail of the generation prompt that insert() held back.
    const std::vector<int> think_prefix = this->enable_think
        ? std::vector<int>{ think_start_id, 198 }                     // <think>\n
        : std::vector<int>{ think_start_id, 271, think_end_id, 271 }; // <think>\n\n</think>\n\n
    if (this->total_tokens + think_prefix.size() >= decode_cap) {
        header_print("WARNING", "Max length reached, stopping generation...");
        meta_info.stop_reason = MAX_LENGTH_REACHED;
        return result;
    }
    // The first prefix forward is left out of DECODING_TIME, as Qwen3_5VL does,
    // so `flm bench` measures decode speed the same way as for qwen3.5:0.8b.
    buffer<bf16> y;
    for (size_t i = 0; i < think_prefix.size(); i++) {
        this->token_history.push_back(think_prefix[i]);
        if (i > 0) this->profiler_list[DECODING_TIME].start();
        y = this->lm_engine->forward(think_prefix[i]);
        if (i > 0) this->profiler_list[DECODING_TIME].stop(1);
        this->total_tokens++;
    }
    if (this->enable_think) {
        // Echo the opening tag so the stream parser enters reasoning mode.
        for (int id : think_prefix) {
            std::string token_str = this->tokenizer->run_time_decoder(id);
            result += token_str;
            os << token_str << std::flush;
        }
    }
    // First real token: the model would open a tool call right here, so it
    // needs the mask too.
    this->_apply_tool_choice_mask(y, meta_info);
    int sampled_token = this->sampler->sample(y);

    while (true) {
        this->profiler_list[TKOEN_DECODE_TIME].start();
        if (this->is_normal_token(sampled_token)){ // filter out special tokens
            std::string token_str = this->tokenizer->run_time_decoder(sampled_token);
            os << token_str << std::flush;
            result += token_str;
        }
        this->profiler_list[TKOEN_DECODE_TIME].stop(1);
        this->token_history.push_back(sampled_token);
        meta_info.generated_tokens++;

        if (this->is_eos(sampled_token)){
            // Keep the kv cache aligned with token_history for a following turn.
            if (this->forward_on_eos &&
                (!this->backend_ || this->backend_->forwards_past_eos()) &&
                this->total_tokens < decode_cap) {
                this->lm_engine->forward(sampled_token);
                this->total_tokens++;
            }
            break;
        }
        if ((length_limit > 0) && (meta_info.generated_tokens >= length_limit)){
            reason = MAX_LENGTH_REACHED;
            break;
        }
        if (this->total_tokens >= decode_cap){
            header_print("WARNING", "Max length reached, stopping generation...");
            reason = MAX_LENGTH_REACHED;
            break;
        }
        if (is_cancelled()) {
            reason = CANCEL_DETECTED;
            // reset stream content
            buffer_.clear();
            current_mode_ = StreamEventType::CONTENT;
            tool_name_.clear();
            is_in_tool_block_ = false;
            break;
        }

        this->profiler_list[DECODING_TIME].start();
        y = this->lm_engine->forward(sampled_token);
        this->profiler_list[DECODING_TIME].stop(1);
        this->total_tokens++;

        this->profiler_list[SAMPLING_TIME].start();
        this->_apply_tool_choice_mask(y, meta_info);
        sampled_token = this->sampler->sample(y);
        this->profiler_list[SAMPLING_TIME].stop(1);
    }
    meta_info.decoding_duration = (uint64_t)(time_utils::cast_to_us(this->profiler_list[DECODING_TIME].get_total_time()).first) * 1e3;
    meta_info.stop_reason = reason;

    std::cout << std::endl;
    if (this->log_raw_output) {
        header_print("FLM", "Model RAW Output: \n" + result);
    }
    return result;
}

std::string MiniCPM_V::generate_with_prompt(chat_meta_info_t& meta_info, lm_uniform_input_t& input, int length_limit, std::ostream& os) {
    if (!this->insert(meta_info, input)) {
        return "";
    }
    // Not _shared_generate: insert() held back the think prefix, which only
    // decode() knows to feed.
    return this->generate(meta_info, length_limit, os);
}

// Non-stream
NonStreamResult MiniCPM_V::parse_nstream_content(const std::string response_text) {
    NonStreamResult result;

    const std::string think_start_tag = "<think>";
    const std::string think_end_tag = "</think>";
    std::string start_tag = "<tool_call>";
    std::string end_tag = "</tool_call>";
    std::string func_end_tag = "</function>";
    std::string func_open = "<function=";
    std::string param_open = "<parameter=";
    std::string param_close = "</parameter>";

    auto trim_tool_value = [](std::string value) {
        while (!value.empty() && (value.front() == '\n' || value.front() == '\r' || value.front() == ' ' || value.front() == '\t')) {
            value.erase(0, 1);
        }
        while (!value.empty() && (value.back() == '\n' || value.back() == '\r' || value.back() == ' ' || value.back() == '\t')) {
            value.pop_back();
        }
        return value;
    };

    // Split off the reasoning block, if any; the rest is the answer.
    std::string answer_text = response_text;
    size_t think_end_pos = response_text.find(think_end_tag);
    if (think_end_pos != std::string::npos) {
        size_t think_start_pos = response_text.find(think_start_tag);
        size_t start = (think_start_pos != std::string::npos && think_start_pos < think_end_pos)
            ? think_start_pos + think_start_tag.length()
            : 0;
        result.reasoning_content = trim_tool_value(response_text.substr(start, think_end_pos - start));
        answer_text = response_text.substr(think_end_pos + think_end_tag.length());
    }

    size_t search_from = 0;

    while (true) {
        size_t start_pos = answer_text.find(start_tag, search_from);
        if (start_pos == std::string::npos) break;

        size_t block_content_start = start_pos + start_tag.length();
        size_t end_pos = answer_text.find(end_tag, block_content_start);

        size_t block_end;
        if (end_pos != std::string::npos) {
            block_end = end_pos;
            search_from = end_pos + end_tag.length();
        } else {
            // Unclosed tag — search for </function> fallback
            size_t func_end_pos = answer_text.find(func_end_tag, block_content_start);
            if (func_end_pos != std::string::npos) {
                block_end = func_end_pos + func_end_tag.length();
            } else {
                block_end = answer_text.length();
            }
            search_from = block_end;
        }

        std::string block = answer_text.substr(block_content_start, block_end - block_content_start);

        std::string tool_name;
        size_t func_start = block.find(func_open);
        if (func_start != std::string::npos) {
            func_start += func_open.length();
            size_t func_name_end = block.find(">", func_start);
            if (func_name_end != std::string::npos) {
                tool_name = block.substr(func_start, func_name_end - func_start);
            }
        }

        nlohmann::json args = nlohmann::json::object();
        size_t pos = 0;

        while (true) {
            size_t param_start = block.find(param_open, pos);
            if (param_start == std::string::npos) break;

            param_start += param_open.length();
            size_t param_name_end = block.find(">", param_start);
            if (param_name_end == std::string::npos) break;

            std::string param_name = block.substr(param_start, param_name_end - param_start);
            size_t value_start = param_name_end + 1;
            size_t value_end = block.find(param_close, value_start);

            size_t next_param_pos = block.find(param_open, value_start);
            size_t func_boundary_pos = block.find(func_end_tag, value_start);

            auto use_earlier_boundary = [&value_end](size_t boundary_pos) {
                if (boundary_pos != std::string::npos && (value_end == std::string::npos || boundary_pos < value_end)) {
                    value_end = boundary_pos;
                }
            };

            use_earlier_boundary(next_param_pos);
            use_earlier_boundary(func_boundary_pos);

            if (value_end == std::string::npos) {
                value_end = block.length();
            }

            std::string param_value = trim_tool_value(block.substr(value_start, value_end - value_start));

            try {
                args[param_name] = nlohmann::json::parse(param_value);
            }
            catch (...) {
                args[param_name] = param_value;
            }

            pos = value_end;
            if (block.compare(value_end, param_close.length(), param_close) == 0) {
                pos += param_close.length();
            }
        }

        result.tool_calls_list.emplace_back(tool_name, args.dump());
    }

    if (result.tool_calls_list.empty()) {
        result.content = trim_tool_value(answer_text);
    } else {
        // Populate legacy single-tool fields from the first call for backward compatibility
        result.tool_name = result.tool_calls_list[0].first;
        result.tool_args = result.tool_calls_list[0].second;
        // Extract content before the first <tool_call>
        size_t first_tool = answer_text.find(start_tag);
        if (first_tool != std::string::npos && first_tool > 0) {
            result.content = trim_tool_value(answer_text.substr(0, first_tool));
        }
    }

    return result;
}

// Stream
StreamResult MiniCPM_V::parse_stream_content(const std::string content) {
    return parse_stream_content_impl(content, false);
}

StreamResult MiniCPM_V::parse_stream_content_final(const std::string content) {
    return parse_stream_content_impl(content, true);
}

StreamResult MiniCPM_V::parse_stream_content_impl(const std::string content, bool is_final) {
    const std::string MARKER_THINK_START = "<think>";
    const std::string MARKER_THINK_END = "</think>";
    const std::string MARKER_TOOL_START = "<tool_call>";
    const std::string MARKER_TOOL_END = "</tool_call>";
    const std::string MARKER_FUNC_END = "</function>";


    StreamResult result;
    buffer_ += content;

    while (true) {
        if (!is_in_tool_block_) {
            size_t stray_end_pos = buffer_.find(MARKER_TOOL_END);
            if (stray_end_pos != std::string::npos) {
                buffer_.erase(stray_end_pos, MARKER_TOOL_END.length());
            }
        }

        if (!is_in_tool_block_) {
            size_t tool_start_pos = buffer_.find(MARKER_TOOL_START);
            if (tool_start_pos != std::string::npos) {
                if (tool_start_pos > 0) {
                    result.content = buffer_.substr(0, tool_start_pos);
                    result.type = current_mode_;
                    buffer_ = buffer_.substr(tool_start_pos);
                    return result;
                }

                is_in_tool_block_ = true;
                buffer_ = buffer_.substr(MARKER_TOOL_START.length());
                result.type = StreamEventType::WAITING;
                return result;
            }
        }

        // tool calling process
        if (is_in_tool_block_) {
            size_t tool_end_pos = buffer_.find(MARKER_TOOL_END);
            size_t func_end_pos = buffer_.find(MARKER_FUNC_END);

            if (tool_end_pos != std::string::npos || func_end_pos != std::string::npos || (is_final && !buffer_.empty())) {
                size_t actual_end_pos = buffer_.size();
                size_t skip_length = 0;

                if (tool_end_pos != std::string::npos) {
                    actual_end_pos = tool_end_pos;
                    skip_length = MARKER_TOOL_END.length();
                }
                else if (func_end_pos != std::string::npos) {
                    actual_end_pos = func_end_pos;
                    skip_length = MARKER_FUNC_END.length();
                }

                std::string block = buffer_.substr(0, actual_end_pos + skip_length);
                buffer_ = buffer_.substr(actual_end_pos + skip_length);
                is_in_tool_block_ = false;

                try {
                    result.type = StreamEventType::TOOL_DONE;
                    result.tool_id = "call_" + std::to_string(std::time(nullptr));

                    // parse function name
                    std::string func_open = "<function=";
                    size_t func_start = block.find(func_open);
                    if (func_start != std::string::npos) {
                        func_start += func_open.length();
                        size_t func_end = block.find(">", func_start);
                        if (func_end != std::string::npos) {
                            result.tool_name = block.substr(func_start, func_end - func_start);
                        }
                    }

                    // parse parameters
                    nlohmann::json args = nlohmann::json::object();
                    std::string param_open = "<parameter=";
                    std::string param_close = "</parameter>";
                    size_t search_pos = 0;

                    while (true) {
                        size_t p_start = block.find(param_open, search_pos);
                        if (p_start == std::string::npos) break;
                        p_start += param_open.length();
                        size_t p_name_end = block.find(">", p_start);
                        if (p_name_end == std::string::npos) break;
                        std::string param_name = block.substr(p_start, p_name_end - p_start);

                        size_t val_start = p_name_end + 1;
                        if (val_start < block.size() && block[val_start] == '\n') val_start++;

                        size_t param_close_pos = block.find(param_close, val_start);
                        size_t val_end = param_close_pos;

                        size_t next_param_pos = block.find(param_open, val_start);
                        size_t func_boundary_pos = block.find(MARKER_FUNC_END, val_start);
                        size_t tool_boundary_pos = block.find(MARKER_TOOL_END, val_start);

                        auto use_earlier_boundary = [&val_end](size_t boundary_pos) {
                            if (boundary_pos != std::string::npos && (val_end == std::string::npos || boundary_pos < val_end)) {
                                val_end = boundary_pos;
                            }
                        };

                        use_earlier_boundary(next_param_pos);
                        use_earlier_boundary(func_boundary_pos);
                        use_earlier_boundary(tool_boundary_pos);

                        if (val_end == std::string::npos && is_final) {
                            val_end = block.size();
                        }
                        if (val_end == std::string::npos) break;

                        std::string param_value = block.substr(val_start, val_end - val_start);

                        // Enhanced trim: handle multiple newlines or spaces that the model may generate after a parameter
                        while(!param_value.empty() && (param_value.back() == '\n' || param_value.back() == '\r' || param_value.back() == ' ')) {
                            param_value.pop_back();
                        }

                        try {
                            // Try to parse as native JSON type (Integer, Float, Boolean, Array, Object)
                            args[param_name] = nlohmann::json::parse(param_value);
                        }
                        catch (...) {
                            args[param_name] = param_value;
                        }

                        search_pos = param_close_pos != std::string::npos && val_end == param_close_pos
                            ? val_end + param_close.length()
                            : val_end;
                    }
                    result.tool_args_str = args.dump();
                    return result;
                }
                catch (...) {
                    result.type = StreamEventType::CONTENT;
                    result.content = "[Error parsing tool call]";
                    return result;
                }
            }
            else {
                result.type = StreamEventType::WAITING;
                return result;
            }
        }

        if (current_mode_ == StreamEventType::CONTENT) {
            size_t think_start_pos = buffer_.find(MARKER_THINK_START);
            if (think_start_pos != std::string::npos) {
                if (think_start_pos > 0) {
                    result.content = buffer_.substr(0, think_start_pos);
                    result.type = StreamEventType::CONTENT;
                    buffer_ = buffer_.substr(think_start_pos);
                    return result;
                }
                buffer_ = buffer_.substr(MARKER_THINK_START.length());
                current_mode_ = StreamEventType::REASONING;
                continue;
            }
        }
        else if (current_mode_ == StreamEventType::REASONING) {
            size_t think_end_pos = buffer_.find(MARKER_THINK_END);
            if (think_end_pos != std::string::npos) {
                if (think_end_pos > 0) {
                    result.content = buffer_.substr(0, think_end_pos);
                    result.type = StreamEventType::REASONING;
                    buffer_ = buffer_.substr(think_end_pos);
                    return result;
                }
                buffer_ = buffer_.substr(MARKER_THINK_END.length());
                current_mode_ = StreamEventType::CONTENT;
                continue;
            }
        }

        if (!buffer_.empty()) {
            size_t last_lt = buffer_.rfind('<');
            // If '<' appears at the end (possibly an incomplete <tool_call> or <think> tag)
            if (last_lt != std::string::npos && (buffer_.length() - last_lt) <= 15) {
                if (last_lt > 0) {
                    // Only output the content before '<'
                    result.content = buffer_.substr(0, last_lt);
                    result.type = current_mode_;
                    buffer_ = buffer_.substr(last_lt);
                    return result;
                } else {
                    // If '<' is the first character in the buffer, directly wait for the next chunk
                    result.type = StreamEventType::WAITING;
                    return result;
                }
            }

            result.content = buffer_;
            result.type = current_mode_;
            buffer_.clear();
            return result;
        }

        break;
    }

    result.type = current_mode_;
    return result;
}
