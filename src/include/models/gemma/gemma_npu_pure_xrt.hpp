/// \file gemma_npu_pure_xrt.hpp
/// \brief gemma_npu_pure_xrt class — native/pure XRT variant of gemma_npu.
/// \author FastFlowLM Team
/// \note Identical interface to gemma_npu; the hot decode path dispatches the
///       per-layer native xrt::run objects directly (pure_xrt_dispatch.hpp)
///       instead of the batched flm_rt (xrt::runlist) layer. XRT-backend mirror
///       of gemma_npu_pure_hrx.
#pragma once
#include "lm_config.hpp"
#include "npu_utils/npu_utils.hpp"
#include "tensor_utils/q4_npu_eXpress.hpp"
#include "models/gemma/gemma_npu_sequence.hpp"
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


class gemma_npu_pure_xrt : public causal_lm{
public:
    gemma_npu_pure_xrt(LM_Config config, npu_xclbin_manager *npu_instance, int MAX_L = 4096);
    ~gemma_npu_pure_xrt();

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
