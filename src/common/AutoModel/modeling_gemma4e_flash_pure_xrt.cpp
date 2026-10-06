/// \file modeling_gemma4e_flash_pure_xrt.cpp
/// \brief Gemma4e_FlashPureXrt implementation. See modeling_gemma4e.cpp for the
///        baseline Gemma4e_Flash; this wrapper only swaps the engine to
///        gemma4e_flash_pure_xrt (native HRX dispatch on the decode path).

#include "AutoModel/modeling_gemma4e_flash_pure_xrt.hpp"


void Gemma4e_FlashPureXrt::create_engine() {
    // native/pure XRT engine variant of gemma4e_flash
    this->lm_engine = std::make_unique<gemma4e_flash_pure_xrt>(*this->lm_config, this->npu.get(), this->MAX_L);
}
