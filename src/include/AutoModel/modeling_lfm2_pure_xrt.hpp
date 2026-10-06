/// \file modeling_lfm2_pure_xrt.hpp
/// \brief Lfm2PureXrt wrapper: identical behaviour to LFM2 but backed by the
///        lfm2_npu_pure_xrt engine (native HRX dispatch instead of the flm_rt
///        hrx::run/runlist shim).
/// \note  Only load_model differs; generation, insert and parsing are inherited
///        (the base LFM2::insert does not cast to the concrete engine type).
#pragma once
#include "AutoModel/modeling_lfm2.hpp"
#include "models/lfm2/lfm2_npu_pure_xrt.hpp"


/************              LFM2 (pure/native HRX)            **************/
class Lfm2PureXrt : public LFM2 {
public:
    Lfm2PureXrt(flm_rt::device* npu_device_inst);

    void load_model(std::string model_path, json model_inf, int default_context_length = -1, bool enable_preemption = false) override;
};
