/// \file d3_npu.hpp
/// \brief d3_npu: the Decision 3.0 (vllm-sr/d3-*) encoder on the NPU
/// \note  The Qwen3.5 text tower with bidirectional full attention, run over packed batches of
///        independent prompts; returns final-norm hidden states for the 255-way code readout
///        that the application applies (FastFlowLM AutoDecisionModel).
#pragma once
#include "lm_config.hpp"
#include "npu_utils/npu_utils.hpp"
#include "tensor_utils/q4_npu_eXpress.hpp"
#include "decision_model.hpp"

/// One image of a request: its patch grid after the processor's resize (16-pixel patches; the
/// image takes grid_h * grid_w / 4 input tokens after the 2 x 2 merge).
struct d3_image {
    int grid_h = 0;
    int grid_w = 0;
};

/// encode_batch()'s `payload` for image requests: the request's images, shared by every prompt.
/// Each prompt holds, in image order, one run of grid_h * grid_w / 4 `image_token_id` tokens per
/// image (the expanded <|image_pad|> placeholders).
struct d3_image_payload {
    std::vector<d3_image> images;
    /// every image's patches in order, as the Qwen2-VL processor flattens them: per patch
    /// [3 channels][2 frames][16][16] bf16, rescaled and normalized; patches in 2 x 2 merge order
    std::vector<bf16> pixels;
    int image_token_id = 248056;
};

class d3_npu : public decision_model{
public:
    /// \param config the package's config.json
    /// \param npu_instance the xclbin manager (mm.xclbin is registered from it)
    /// \param max_length longest prompt accepted, in tokens (default: 32,768, FastFlowLM's limit)
    d3_npu(LM_Config config, npu_xclbin_manager* npu_instance, int max_length = 32768);
    ~d3_npu();
    d3_npu(const d3_npu&) = delete;
    d3_npu& operator=(const d3_npu&) = delete;

    void load_weights(Q4NX& q4nx) override;
    /// \param payload nullptr (text) or a d3_image_payload*: the vision tower runs once per call and
    ///        its output fills every prompt's image tokens; positions are Qwen3.5's 3D M-RoPE ones.
    ///        The vision bitstreams are registered on the first call with images.
    std::vector<buffer<bf16>> encode_batch(const std::vector<std::vector<int>>& ids, void* payload = nullptr) override;
    buffer<bf16> encode(const std::vector<int>& ids, void* payload = nullptr) override;
    int max_length() override;
    int hidden_size() override;

    /// \brief rows packed into one pass (default 2048, four array GEMM row tiles); 0 = one prompt per pass
    void set_pack_rows(int rows);
    /// \brief rig only: run the full-attention layers causally (the stock-engine comparison baseline)
    void set_causal(bool causal);

private:
    struct Impl;
    Impl* _impl;
};
