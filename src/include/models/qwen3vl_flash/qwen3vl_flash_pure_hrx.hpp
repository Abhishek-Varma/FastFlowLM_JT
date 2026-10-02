/// \file qwen3vl_flash_pure_hrx.hpp
/// \brief qwen3vl_flash_pure_hrx class
/// \author FastFlowLM Team
/// \date 2026-09-30
/// \version 0.9.28
/// \note Experimental Qwen3-VL flash engine identical to qwen3vl_flash except
///       that its hot decode path (decoder-layer chain + lm_head) calls the
///       native/pure HRX C API (hrx_stream_dispatch / hrx_stream_flush /
///       hrx_stream_wait) directly instead of going through the flm_rt
///       (hrx::run / hrx::runlist) C++ shim. The prefill path is unchanged.
///       Used to benchmark the shim overhead against a native-dispatch path.
#pragma once
#include "models/qwen3vl/qwen3vl_npu.hpp"


class qwen3vl_flash_pure_hrx : public causal_lm{
public:
    /// \brief  initialize the qwen3vl_flash_pure_hrx
    /// \param config the configuration
    /// \param npu_instance the npu instance
    qwen3vl_flash_pure_hrx(LM_Config config, npu_xclbin_manager *npu_instance, int MAX_L = 4096);
    ~qwen3vl_flash_pure_hrx();

    /// \brief forward the qwen3vl_flash_pure_hrx
    /// \param ids the ids
    /// \return the output tensor
    buffer<bf16> forward(int ids) override;
    buffer<bf16> prefill(std::vector<int>& ids, void* payload = nullptr) override;

    /// \brief set the context length
    /// \param L the context length
    void set_context_length(int L) override;

    /// \brief load the weights
    /// \param q4nx the q4nx
    void load_weights(Q4NX& q4nx) override;

    /// \brief update the max length
    void clear_context() override;

    /// \brief get the k cache
    /// \param layer_idx the layer index
    /// \param idx the index
    /// \return the k cache
    buffer<bf16> get_k_cache(int layer_idx, int idx) override;

    /// \brief get the v cache
    /// \param layer_idx the layer index
    /// \param idx the index
    /// \return the v cache
    buffer<bf16> get_v_cache(int layer_idx, int idx) override;

    /// \brief update the max length
    /// \param MAX_L the max length
    void update_max_length(uint32_t MAX_L) override;

    /// \brief get the current context length
    /// \return the current context length
    int get_current_context_length() override;
    int checkpoint() override;
    int restore() override;
private:
    struct Impl;
    Impl* _impl;
};
