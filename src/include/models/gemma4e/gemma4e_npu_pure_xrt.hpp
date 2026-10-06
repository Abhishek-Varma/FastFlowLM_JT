/// \file gemma4e_npu_pure_xrt.hpp
/// \brief gemma4e_npu_pure_xrt class — native/pure XRT variant of gemma4e_npu.
/// \author FastFlowLM Team
/// \note Identical interface and preprocessing to gemma4e_npu; the hot decode path
///       (decoder-layer chain + lm_head + background preload) dispatches via the raw
///       HRX C API (pure_xrt_dispatch.hpp) instead of the flm_rt run/runlist shim.
///       Shared image/audio typedefs + enum helpers come from the stock gemma4e_npu.hpp.
#pragma once
#include "models/gemma4e/gemma4e_npu.hpp"


class gemma4e_npu_pure_xrt : public causal_lm{
public:
    gemma4e_npu_pure_xrt(LM_Config config, npu_xclbin_manager *npu_instance, int MAX_L = 4096);
    ~gemma4e_npu_pure_xrt();

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

    // parameters for vision preprocessing in Gemma4e (mirror gemma4e_npu; read by the
    // image/audio encoders through their parent pointer)
    unsigned int GEMMA4E_VISION_MAX_POSITION_EMBEDDINGS;
    unsigned int GEMMA4E_VISION_NUM_HIDDEN_LAYERS;
    unsigned int GEMMA4E_VISION_NUM_ATTENTION_HEADS;
    unsigned int GEMMA4E_VISION_HIDDEN_SIZE;
    unsigned int GEMMA4E_VISION_INTERMEDIATE_SIZE;
    unsigned int GEMMA4E_VISION_HEAD_DIM;
    unsigned int GEMMA4E_VISION_PATCH_SIZE;
    float GEMMA4E_ROPE_THETA;
    unsigned int GEMMA4E_POOLING_KERNEL_SIZE;
    unsigned int GEMMA4E_POSITION_EMBEDDING_SIZE;
    unsigned int GEMMA4E_VISION_IMAGE_OUTPUT_SIZE;
    float GEMMA4E_VISION_RESCALE_FACTOR;
    float GEMMA4E_VISION_IMAGE_MEAN;
    float GEMMA4E_VISION_IMAGE_STD;

    // parameters for audio preprocessing in Gemma4e
    unsigned int Audio_MM_TILE_M;
    unsigned int Audio_MM_TILE_K;
    unsigned int Audio_MM_TILE_N;
    int Gemma4E_Audio_resample_rate;
    float Gemma4E_Audio_gradient_clipping;
    unsigned int Gemma4E_Audio_Multimodal_Output_SIZE;
    unsigned int Gemma4E_Audio_language_projection_output_size;
    unsigned int Gemma4E_Audio_HIDDEN_SIZE;
    unsigned int Gemma4E_Audio_INTERMEDIATE_SIZE;
    unsigned int Gemma4E_Audio_attention_chunk_size;
    unsigned int Gemma4E_Audio_attention_context_left;
    unsigned int Gemma4E_Audio_attention_context_right;
    unsigned int Gemma4E_Audio_num_attention_heads;
    unsigned int Gemma4E_Audio_num_attention_layers;
    unsigned int Gemma4E_Audio_conv1d_kernel_size;
    unsigned int Gemma4E_Audio_conv1d_stride;
    unsigned int Gemma4E_Audio_conv2d_kernel_size;
    unsigned int Gemma4E_Audio_conv2d_Stride;
    unsigned int Gemma4e_Audio_conv2d_Padding;
    unsigned int Gemma4E_Audio_subsampling_conv_channels_0;
    unsigned int Gemma4E_Audio_subsampling_conv_channels_1;
    float Gemma4E_Audio_attention_softcap;


    inline void load_vision_preprocess_parameters(LM_Config& config){
        const nlohmann::json& vc = config.sub("vision_config");
        GEMMA4E_VISION_MAX_POSITION_EMBEDDINGS = vc.value("GEMMA4E_VISION_MAX_POSITION_EMBEDDINGS", -1);
        GEMMA4E_VISION_NUM_HIDDEN_LAYERS   = vc.value("GEMMA4E_VISION_NUM_HIDDEN_LAYERS", -1);
        GEMMA4E_VISION_NUM_ATTENTION_HEADS = vc.value("GEMMA4E_VISION_NUM_ATTENTION_HEADS", -1);
        GEMMA4E_VISION_HIDDEN_SIZE         = vc.value("GEMMA4E_VISION_HIDDEN_SIZE", -1);
        GEMMA4E_VISION_INTERMEDIATE_SIZE   = vc.value("GEMMA4E_VISION_INTERMEDIATE_SIZE", -1);
        GEMMA4E_VISION_HEAD_DIM            = vc.value("GEMMA4E_VISION_HEAD_DIM", -1);
        GEMMA4E_VISION_PATCH_SIZE          = vc.value("GEMMA4E_VISION_PATCH_SIZE", -1);
        GEMMA4E_ROPE_THETA                 = vc.value("GEMMA4E_ROPE_THETA", -1.0f);
        GEMMA4E_POOLING_KERNEL_SIZE        = vc.value("GEMMA4E_POOLING_KERNEL_SIZE", -1);
        GEMMA4E_POSITION_EMBEDDING_SIZE    = vc.value("GEMMA4E_POSITION_EMBEDDING_SIZE", -1);
        GEMMA4E_VISION_IMAGE_OUTPUT_SIZE   = vc.value("GEMMA4E_VISION_IMAGE_OUTPUT_SIZE", -1);
        GEMMA4E_VISION_RESCALE_FACTOR      = vc.value("GEMMA4E_VISION_RESCALE_FACTOR", -1.0f);
        GEMMA4E_VISION_IMAGE_MEAN          = vc.value("GEMMA4E_VISION_IMAGE_MEAN", -1.0f);
        GEMMA4E_VISION_IMAGE_STD           = vc.value("GEMMA4E_VISION_IMAGE_STD", -1.0f);
    }

    inline void load_audio_preprocess_parameters(LM_Config& config){
        const nlohmann::json& ac = config.sub("audio_config");
        Audio_MM_TILE_M = ac.value("Audio_MM_TILE_M", 128);
        Audio_MM_TILE_K = ac.value("Audio_MM_TILE_K", 512);
        Audio_MM_TILE_N = ac.value("Audio_MM_TILE_N", 64);
        Gemma4E_Audio_resample_rate = ac.value("Gemma4E_Audio_audio_resample_rate", -1);
        Gemma4E_Audio_gradient_clipping = ac.value("Gemma4E_Audio_gradient_clipping", -1.0f);
        Gemma4E_Audio_Multimodal_Output_SIZE = ac.value("Gemma4E_Audio_Multimodal_Output_SIZE", -1);
        Gemma4E_Audio_language_projection_output_size = ac.value("Gemma4E_Audio_language_projection_output_size", -1);
        Gemma4E_Audio_HIDDEN_SIZE = ac.value("Gemma4E_Audio_HIDDEN_SIZE", -1);
        Gemma4E_Audio_INTERMEDIATE_SIZE = ac.value("Gemma4E_Audio_INTERMEDIATE_SIZE", -1);
        Gemma4E_Audio_attention_chunk_size = ac.value("Gemma4E_Audio_attention_chunk_size", -1);
        Gemma4E_Audio_attention_context_left = ac.value("Gemma4E_Audio_attention_context_left", -1);
        Gemma4E_Audio_attention_context_right = ac.value("Gemma4E_Audio_attention_context_right", -1);
        Gemma4E_Audio_num_attention_heads = ac.value("Gemma4E_Audio_num_attention_heads", -1);
        Gemma4E_Audio_num_attention_layers = ac.value("Gemma4E_Audio_num_attention_layers", -1);
        Gemma4E_Audio_conv1d_kernel_size = ac.value("Gemma4E_Audio_conv1d_kernel_size", -1);
        Gemma4E_Audio_conv1d_stride = ac.value("Gemma4E_Audio_conv1d_stride", -1);
        Gemma4E_Audio_conv2d_kernel_size = ac.value("Gemma4E_conv2d_kernel_size", -1);
        Gemma4E_Audio_conv2d_Stride = ac.value("Gemma4E_conv2d_Stride", -1);
        Gemma4e_Audio_conv2d_Padding = ac.value("Gemma4e_conv2d_Padding", -1);
        Gemma4E_Audio_subsampling_conv_channels_0 = ac.value("Gemma4E_Audio_subsampling_conv_channels_0", -1);
        Gemma4E_Audio_subsampling_conv_channels_1 = ac.value("Gemma4E_Audio_subsampling_conv_channels_1", -1);
        Gemma4E_Audio_attention_softcap = ac.value("Gemma4E_Audio_attention_softcap", -1.0f);
    }

private:
    struct Impl;
    Impl* _impl;
};
