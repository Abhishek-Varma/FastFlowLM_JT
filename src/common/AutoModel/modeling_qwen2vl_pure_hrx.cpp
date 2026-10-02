/// \file modeling_qwen2vl_pure_hrx.cpp
/// \brief Qwen2VLPureHrx implementation. See modeling_qwen2vl.cpp for the
///        baseline Qwen2VL; this wrapper only swaps the engine to
///        qwen2vl_npu_pure_hrx (native HRX dispatch on the decode path).
///        Everything else is inherited from Qwen2VL.

#include "AutoModel/modeling_qwen2vl_pure_hrx.hpp"


void Qwen2VLPureHrx::create_engine() {
    // native/pure HRX engine variant of qwen2vl_npu
    this->lm_engine = std::make_unique<qwen2vl_npu_pure_hrx>(*this->lm_config, this->npu.get(), this->MAX_L);
}
