/// \file modeling_gemma4e_pure_hrx.cpp
/// \brief Gemma4ePureHrx implementation. See modeling_gemma4e.cpp for the baseline
///        Gemma4e; this wrapper only swaps the engine to gemma4e_npu_pure_hrx (native
///        HRX dispatch on the decode path).

#include "AutoModel/modeling_gemma4e_pure_hrx.hpp"


void Gemma4ePureHrx::create_engine() {
    // native/pure HRX engine variant of gemma4e_npu
    this->lm_engine = std::make_unique<gemma4e_npu_pure_hrx>(*this->lm_config, this->npu.get(), this->MAX_L);
}
