/// \file gpt_oss_npu_pure_hrx.hpp
/// \brief gpt_oss_npu_pure_hrx class -- native/pure HRX dispatch variant of gpt_oss_npu.
/// \note  Same public interface as gpt_oss_npu; only the internal dispatch of the
///        hot decode path (per-layer runlist + lm_head) bypasses the flm_rt
///        (hrx::run/runlist) shim in favour of the raw libhrx C API.
#pragma once
#include "lm_config.hpp"
#include "npu_utils/npu_utils.hpp"
#include "tensor_utils/q4_npu_eXpress.hpp"
#include "models/gpt_oss/gpt_oss_npu_sequence.hpp"
#include "modules/embedding.hpp"
#include "modules/lm_head.hpp"
#include "modules/gemm.hpp"
#include "modules/dequant.hpp"
#include "tensor_2d.hpp"
#include "utils/utils.hpp"
#include "causal_lm.hpp"
#if USEAVX2
#include <immintrin.h>  // For AVX intrinsics
#endif


class gpt_oss_npu_pure_hrx : public causal_lm{
public:
    /// \brief  initialize the gpt_oss_npu_pure_hrx
    /// \param config the configuration
    /// \param npu_instance the npu instance
    gpt_oss_npu_pure_hrx(LM_Config config, npu_xclbin_manager *npu_instance, int MAX_L = 4096);
    ~gpt_oss_npu_pure_hrx();

    /// \brief forward the gpt_oss_npu_pure_hrx
    /// \param ids the ids
    /// \return the output tensor
    buffer<bf16> forward(int ids) override;
    buffer<bf16> forward(buffer<bf16>& x);
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
    buffer<bf16> get_k_cache(int layer_idx, int idx) override;

    /// \brief get the v cache
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
