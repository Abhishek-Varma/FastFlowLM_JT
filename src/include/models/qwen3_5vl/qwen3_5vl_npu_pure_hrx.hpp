/// \file qwen3_5vl_npu_pure_hrx.hpp
/// \brief qwen3_5vl_npu_pure_hrx class
/// \author FastFlowLM Team
/// \note Experimental Qwen3.5-VL engine identical to qwen3_5vl_npu except that its
///       hot decode path (decoder-layer chain + lm_head) calls the native/pure
///       HRX C API directly instead of going through the flm_rt (hrx::run /
///       hrx::runlist) C++ shim. The prefill path is unchanged. Used to measure
///       NPU vs host compute via the phase-tagged device-wait accounting.
#pragma once
#include "models/qwen3_5vl/qwen3_5vl_npu.hpp"

class qwen3_5vl_npu_pure_hrx : public causal_lm{
public:
    qwen3_5vl_npu_pure_hrx(LM_Config config, npu_xclbin_manager *npu_instance, int MAX_L = 4096);
    ~qwen3_5vl_npu_pure_hrx();

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

    unsigned int QWEN3_5_PATCH_SIZE;
    unsigned int QWEN3_5_IMAGE_MERGE_SIZE;
    unsigned int QWEN3_5_SPATIAL_MERGE_SIZE;
    unsigned int QWEN3_5_SHORTEST_EDGE;
    unsigned int QWEN3_5_LONGEST_EDGE;
    float QWEN3_5_VISION_RESCALE_FACTOR;
    float QWEN3_5_VISION_RESCALE_IMAGE_MEAN;
    float QWEN3_5_VISION_RESCALE_IMAGE_STD;
    unsigned int QWEN3_5_TEMPORAL_PATCH_SIZE;
    unsigned int QWEN3_5_MERGE_SIZE;

    inline void load_vision_preprocess_parameters(LM_Config& config){
        const nlohmann::json& vc = config.sub("vision_config");
        QWEN3_5_PATCH_SIZE  = vc.value("QWEN3_5_PATCH_SIZE", -1);
        QWEN3_5_IMAGE_MERGE_SIZE = vc.value("QWEN3_5_IMAGE_MERGE_SIZE", -1);
        QWEN3_5_SPATIAL_MERGE_SIZE = vc.value("QWEN3_5_SPATIAL_MERGE_SIZE", -1);
        QWEN3_5_SHORTEST_EDGE = vc.value("QWEN3_5_SHORTEST_EDGE", -1);
        QWEN3_5_LONGEST_EDGE = vc.value("QWEN3_5_LONGEST_EDGE", -1);
        QWEN3_5_VISION_RESCALE_FACTOR = vc.value("QWEN3_5_VISION_RESCALE_FACTOR", -1.0f);
        QWEN3_5_VISION_RESCALE_IMAGE_MEAN = vc.value("QWEN3_5_VISION_RESCALE_IMAGE_MEAN", -1.0f);
        QWEN3_5_VISION_RESCALE_IMAGE_STD = vc.value("QWEN3_5_VISION_RESCALE_IMAGE_STD", -1.0f);
        QWEN3_5_TEMPORAL_PATCH_SIZE = vc.value("QWEN3_5_TEMPORAL_PATCH_SIZE", -1);
        QWEN3_5_MERGE_SIZE = QWEN3_5_IMAGE_MERGE_SIZE;
    }
private:
    struct Impl;
    Impl* _impl;
};
