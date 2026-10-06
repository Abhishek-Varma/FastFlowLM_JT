/// \file modeling_qwen3_5vl_pure_xrt.cpp
/// \brief Qwen3_5VLPureXrt implementation. See modeling_qwen3_5vl.cpp for the
///        baseline Qwen3_5VL; this wrapper only swaps the engine to
///        qwen3_5vl_npu_pure_xrt (native HRX dispatch on the decode path).
///        Everything else is inherited from Qwen3_5VL.

#include "AutoModel/modeling_qwen3_5vl_pure_xrt.hpp"


void Qwen3_5VLPureXrt::create_engine() {
    // native/pure XRT engine variant of qwen3_5vl_npu
    this->lm_engine = std::make_unique<qwen3_5vl_npu_pure_xrt>(*this->lm_config, this->npu.get(), this->MAX_L);
}
