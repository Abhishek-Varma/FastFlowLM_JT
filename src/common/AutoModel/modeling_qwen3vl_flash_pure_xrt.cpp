/// \file modeling_qwen3vl_flash_pure_xrt.cpp
/// \brief Qwen3VL_FlashPureXrt implementation. See modeling_qwen3vl.cpp for the
///        baseline Qwen3VL_Flash; this wrapper only swaps the engine to
///        qwen3vl_flash_pure_xrt (native HRX dispatch on the decode path).
///        Everything else is inherited from Qwen3VL_Flash.

#include "AutoModel/modeling_qwen3vl_flash_pure_xrt.hpp"


void Qwen3VL_FlashPureXrt::create_engine() {
    // native/pure XRT engine variant of qwen3vl_flash
    this->lm_engine = std::make_unique<qwen3vl_flash_pure_xrt>(*this->lm_config, this->npu.get(), this->MAX_L);
}
