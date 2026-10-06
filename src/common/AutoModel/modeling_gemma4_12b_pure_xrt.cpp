/// \file modeling_gemma4_12b_pure_xrt.cpp
/// \brief Gemma4_12BPureXrt implementation. See modeling_gemma4_12b.cpp for the
///        baseline Gemma4_12B; this wrapper only swaps the engine to
///        gemma4_12b_npu_pure_xrt (native HRX dispatch on the decode path).

#include "AutoModel/modeling_gemma4_12b_pure_xrt.hpp"


void Gemma4_12BPureXrt::create_engine() {
    // native/pure XRT engine variant of gemma4_12b_npu
    this->lm_engine = std::make_unique<gemma4_12b_npu_pure_xrt>(*this->lm_config, this->npu.get(), this->MAX_L);
}
