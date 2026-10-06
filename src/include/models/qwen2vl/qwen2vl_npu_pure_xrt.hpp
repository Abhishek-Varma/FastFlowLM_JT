/// \file qwen2vl_npu_pure_xrt.hpp
/// \brief qwen2vl_npu_pure_xrt class
/// \author FastFlowLM Team
/// \date 2026-09-30
/// \version 0.9.28
/// \note Experimental Qwen2-VL engine identical to qwen2vl_npu except that its
///       hot decode path (decoder-layer chain + lm_head) calls the native/pure
///       HRX C API directly instead of the flm_rt (hrx::run / hrx::runlist)
///       shim. The prefill path is unchanged.
#pragma once
#include "models/qwen2vl/qwen2vl_npu.hpp"


class qwen2vl_npu_pure_xrt : public causal_lm{
public:
    qwen2vl_npu_pure_xrt(LM_Config config, npu_xclbin_manager *npu_instance, int MAX_L = 4096);
    ~qwen2vl_npu_pure_xrt();

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
private:
    struct Impl;
    Impl* _impl;
};
