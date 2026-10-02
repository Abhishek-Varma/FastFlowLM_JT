/// \file modeling_qwen3_6_moe_pure_hrx.cpp
/// \brief Qwen3_6_MOEPureHrx implementation. See modeling_qwen3_6_moe.cpp for the
///        baseline Qwen3_6_MOE; this wrapper only swaps the engine to
///        qwen3_6_moe_npu_pure_hrx (native HRX dispatch on the decode path).
///        Everything else is inherited from Qwen3_6_MOE.

#include "AutoModel/modeling_qwen3_6_moe_pure_hrx.hpp"


void Qwen3_6_MOEPureHrx::create_engine() {
    // native/pure HRX engine variant of qwen3_6_moe_npu
    this->lm_engine = std::make_unique<qwen3_6_moe_npu_pure_hrx>(*this->lm_config, this->npu.get(), this->MAX_L);
}
