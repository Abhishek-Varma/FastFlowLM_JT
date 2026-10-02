/// \file gemma4_12b_npu_pure_hrx.hpp
/// \brief gemma4_12b_npu_pure_hrx class — native/pure HRX variant of gemma4_12b_npu.
/// \author FastFlowLM Team
/// \note Identical interface and preprocessing to gemma4_12b_npu; the hot decode path
///       (decoder-layer chain + lm_head + background preload) dispatches via the raw
///       HRX C API (pure_hrx_dispatch.hpp) instead of the flm_rt run/runlist shim.
///       The shared image/audio typedefs come from the stock gemma4_12b_npu.hpp.
#pragma once
#include "models/gemma4_12b/gemma4_12b_npu.hpp"


class gemma4_12b_npu_pure_hrx : public causal_lm{
public:
    gemma4_12b_npu_pure_hrx(LM_Config config, npu_xclbin_manager *npu_instance, int MAX_L = 4096);
    ~gemma4_12b_npu_pure_hrx();

    buffer<bf16> forward(int ids) override;
    buffer<bf16> prefill(std::vector<int>& ids, void* payload = nullptr) override;

    void set_context_length(int L) override;

    void load_weights(Q4NX& q4nx) override;

    void clear_context() override;

    buffer<bf16> get_k_cache(int layer_idx, int idx) override;

    buffer<bf16> get_v_cache(int layer_idx, int idx) override;

    void update_max_length(uint32_t MAX_L) override;

    int get_current_context_length() override;
    int checkpoint() override;
    int restore() override;

    // ---- preprocessing parameters (mirror gemma4_12b_npu; read by Impl) ----------
    unsigned int GEMMA4_12B_vision_pooling_kernel_size;
    unsigned int GEMMA4_12B_vision_patch_size;
    unsigned int GEMMA4_12B_vision_max_soft_tokens;
    float GEMMA4_12B_vision_rescale_factor;
    float GEMMA4_12B_vision_image_mean;
    float GEMMA4_12B_vision_image_std;
    unsigned int GEMMA4_12B_audio_embed_dim;
    unsigned int GEMMA4_12B_audio_samples_per_token;
    unsigned int GEMMA4_12B_audio_max_soft_tokens;
    unsigned int GEMMA4_12B_audio_sampling_rate;

    inline const nlohmann::json& _processor_sub(LM_Config& config, const char* key){
        return cfg_sub(cfg_sub(config._json_config, "processor_config"), key);
    }

    template <typename T>
    inline T _preprocess_value(const nlohmann::json& processor, const nlohmann::json& fallback,
                              const char* key, T default_value){
        return cfg_get<T>(processor, key, cfg_get<T>(fallback, key, default_value));
    }

    inline float _preprocess_channel_value(const nlohmann::json& processor, const nlohmann::json& fallback,
                                           const char* key, float default_value){
        for (const nlohmann::json* jc : {&processor, &fallback}){
            if (!jc->contains(key) || (*jc)[key].is_null()){
                continue;
            }
            const nlohmann::json& v = (*jc)[key];
            if (!v.is_array()){
                return float(v);
            }
            if (v.empty()){
                continue;
            }
            const float first = float(v[0]);
            for (const auto& c : v){
                if (float(c) != first){
                    header_print("warning", std::string("processor config ") + key
                                 + " is not uniform across channels, using " + std::to_string(first));
                    break;
                }
            }
            return first;
        }
        return default_value;
    }

    inline void load_vision_preprocess_parameters(LM_Config& config){
        const nlohmann::json& pc = this->_processor_sub(config, "image_processor");
        const nlohmann::json& vc = config.sub("vision_config");
        GEMMA4_12B_vision_pooling_kernel_size = _preprocess_value<unsigned int>(pc, vc, "pooling_kernel_size", 3);
        GEMMA4_12B_vision_patch_size          = _preprocess_value<unsigned int>(pc, vc, "patch_size", 16);
        GEMMA4_12B_vision_max_soft_tokens     = _preprocess_value<unsigned int>(pc, vc, "max_soft_tokens", 280);
        GEMMA4_12B_vision_rescale_factor      = _preprocess_value<float>(pc, vc, "rescale_factor", 1.0f / 255.0f);
        GEMMA4_12B_vision_image_mean          = _preprocess_channel_value(pc, vc, "image_mean", 0.0f);
        GEMMA4_12B_vision_image_std           = _preprocess_channel_value(pc, vc, "image_std", 1.0f);
    }

    inline void load_audio_preprocess_parameters(LM_Config& config){
        const nlohmann::json& pc = this->_processor_sub(config, "feature_extractor");
        const nlohmann::json& ac = config.sub("audio_config");
        GEMMA4_12B_audio_embed_dim          = _preprocess_value<unsigned int>(pc, ac, "feature_size", 640);
        GEMMA4_12B_audio_samples_per_token  = _preprocess_value<unsigned int>(pc, ac, "audio_samples_per_token", 640);
        GEMMA4_12B_audio_sampling_rate      = _preprocess_value<unsigned int>(pc, ac, "sampling_rate", 16000);
        GEMMA4_12B_audio_max_soft_tokens    = cfg_get<unsigned int>(
            cfg_sub(config._json_config, "processor_config"), "audio_seq_length",
            cfg_get<unsigned int>(ac, "audio_seq_length", 750));
    }

private:
    struct Impl;
    Impl* _impl;
};
