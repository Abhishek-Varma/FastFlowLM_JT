/// \file modeling_gemma3_pure_xrt.cpp
/// \brief Gemma3PureXrt implementation. See modeling_gemma3.cpp for the baseline
///        Gemma3; this wrapper only swaps the engine to gemma_npu_pure_xrt (native
///        XRT per-run dispatch on the decode path). XRT-backend mirror of
///        modeling_gemma3_pure_hrx.cpp. Everything else is inherited.

#include "AutoModel/modeling_gemma3_pure_xrt.hpp"


void Gemma3PureXrt::create_engine() {
    // native/pure XRT engine variant of gemma_npu
    this->lm_engine = std::make_unique<gemma_npu_pure_xrt>(*this->lm_config, this->npu.get(), this->MAX_L);
}
