/// \file modeling_qwen3vl_pure_hrx.cpp
/// \brief Qwen3VLPureHrx implementation. See modeling_qwen3vl.cpp for the
///        baseline Qwen3VL; this wrapper only swaps the engine to
///        qwen3vl_npu_pure_hrx (native HRX dispatch on the decode path).
///        Everything else is inherited from Qwen3VL.

#include "AutoModel/modeling_qwen3vl_pure_hrx.hpp"


void Qwen3VLPureHrx::create_engine() {
    // native/pure HRX engine variant of qwen3vl_npu
    this->lm_engine = std::make_unique<qwen3vl_npu_pure_hrx>(*this->lm_config, this->npu.get(), this->MAX_L);
}
