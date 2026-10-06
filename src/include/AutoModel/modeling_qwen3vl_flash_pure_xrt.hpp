/// \file modeling_qwen3vl_flash_pure_xrt.hpp
/// \brief Qwen3VL_FlashPureXrt wrapper: identical behaviour to Qwen3VL_Flash but
///        backed by the qwen3vl_flash_pure_xrt engine, whose hot decode path
///        calls the native HRX C API directly instead of the flm_rt
///        (hrx::run/runlist) shim.
/// \note  Only create_engine (engine instantiation) differs from Qwen3VL_Flash;
///        generation, parsing, vision preprocessing and the single-turn
///        checkpoint logic are inherited unchanged.
#pragma once
#include "AutoModel/modeling_qwen3vl.hpp"
#include "models/qwen3vl_flash/qwen3vl_flash_pure_xrt.hpp"


/************          qwen3vl_flash (pure/native HRX)          **************/
class Qwen3VL_FlashPureXrt : public Qwen3VL_Flash {
protected:
    void create_engine() override;

public:
    Qwen3VL_FlashPureXrt(flm_rt::device* npu_device_inst) : Qwen3VL_Flash(npu_device_inst) {}
};
