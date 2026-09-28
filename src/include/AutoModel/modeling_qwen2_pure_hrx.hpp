/// \file modeling_qwen2_pure_hrx.hpp
/// \brief Qwen2PureHrx wrapper: identical behaviour to Qwen2 but backed by the
///        qwen2_npu_pure_hrx engine (native HRX dispatch instead of the flm_rt
///        hrx::run/runlist shim).
/// \note  Only load_model differs; generation, insert and parsing are inherited
///        (the base insert does not cast to the concrete engine type).
#pragma once
#include "AutoModel/modeling_qwen2.hpp"
#include "models/qwen2/qwen2_npu_pure_hrx.hpp"


/************              qwen2 (pure/native HRX)            **************/
class Qwen2PureHrx : public Qwen2 {
public:
    Qwen2PureHrx(flm_rt::device* npu_device_inst);

    void load_model(std::string model_path, json model_inf, int default_context_length = -1, bool enable_preemption = false) override;
};
