/// \file modeling_phi4_pure_hrx.hpp
/// \brief Phi4PureHrx wrapper: identical behaviour to Phi4 but backed by the
///        phi4_npu_pure_hrx engine (native HRX dispatch instead of the flm_rt
///        hrx::run/runlist shim).
/// \note  Only load_model differs; generation, insert and parsing are inherited
///        (the base insert does not cast to the concrete engine type).
#pragma once
#include "AutoModel/modeling_phi4.hpp"
#include "models/phi4/phi4_npu_pure_hrx.hpp"


/************              phi4 (pure/native HRX)            **************/
class Phi4PureHrx : public Phi4 {
public:
    Phi4PureHrx(flm_rt::device* npu_device_inst);

    void load_model(std::string model_path, json model_inf, int default_context_length = -1, bool enable_preemption = false) override;
};
