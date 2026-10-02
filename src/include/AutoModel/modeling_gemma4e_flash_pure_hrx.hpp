/// \file modeling_gemma4e_flash_pure_hrx.hpp
/// \brief Gemma4e_FlashPureHrx wrapper: identical behaviour to Gemma4e_Flash but
///        backed by the gemma4e_flash_pure_hrx engine, whose hot decode path calls
///        the native HRX C API directly instead of the flm_rt (hrx::run/runlist) shim.
/// \note  Only create_engine differs from Gemma4e_Flash; generation, parsing and
///        image/audio preprocessing are inherited unchanged.
#pragma once
#include "AutoModel/modeling_gemma4e.hpp"
#include "models/gemma4e_flash/gemma4e_flash_pure_hrx.hpp"


/************        gemma4e_flash (pure/native HRX)        **************/
class Gemma4e_FlashPureHrx : public Gemma4e_Flash {
protected:
    void create_engine() override;

public:
    Gemma4e_FlashPureHrx(flm_rt::device* npu_device_inst) : Gemma4e_Flash(npu_device_inst) {}
};
