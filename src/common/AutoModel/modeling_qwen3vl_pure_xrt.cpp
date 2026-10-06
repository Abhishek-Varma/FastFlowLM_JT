/// \file modeling_qwen3vl_pure_xrt.cpp
/// \brief Qwen3VLPureXrt implementation. See modeling_qwen3vl.cpp for the
///        baseline Qwen3VL; this wrapper only swaps the engine to
///        qwen3vl_npu_pure_xrt (native HRX dispatch on the decode path).
///        Everything else is inherited from Qwen3VL.

#include "AutoModel/modeling_qwen3vl_pure_xrt.hpp"


void Qwen3VLPureXrt::create_engine() {
    // native/pure XRT engine variant of qwen3vl_npu
    this->lm_engine = std::make_unique<qwen3vl_npu_pure_xrt>(*this->lm_config, this->npu.get(), this->MAX_L);
}
