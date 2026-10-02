/// \file modeling_gemma3_pure_hrx.cpp
/// \brief Gemma3PureHrx implementation. See modeling_gemma3.cpp for the baseline
///        Gemma3; this wrapper only swaps the engine to gemma_npu_pure_hrx (native
///        HRX dispatch on the decode path). Everything else is inherited.

#include "AutoModel/modeling_gemma3_pure_hrx.hpp"


void Gemma3PureHrx::create_engine() {
    // native/pure HRX engine variant of gemma_npu
    this->lm_engine = std::make_unique<gemma_npu_pure_hrx>(*this->lm_config, this->npu.get(), this->MAX_L);
}
