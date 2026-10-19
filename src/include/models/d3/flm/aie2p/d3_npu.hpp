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

class d3_npu : public decision_model{
public:
    /// \param config the package's config.json
    /// \param npu_instance the xclbin manager (mm.xclbin is registered from it)
    /// \param max_length longest prompt accepted, in tokens
    d3_npu(LM_Config config, npu_xclbin_manager* npu_instance, int max_length = 4096);
    ~d3_npu();
    d3_npu(const d3_npu&) = delete;
    d3_npu& operator=(const d3_npu&) = delete;

    void load_weights(Q4NX& q4nx) override;
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
