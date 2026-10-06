/// \file modeling_llama3_pure_xrt.hpp
/// \brief Llama3PureXrt wrapper: identical behaviour to Llama3 but backed by the
///        llama_npu_pure_xrt engine, whose hot dispatch paths call the native HRX
///        C API directly instead of the flm_rt (hrx::run/runlist) shim.
/// \note  Only load_model (engine instantiation) differs from Llama3; generation,
///        insert and parsing are inherited unchanged (the base insert does not
///        cast to the concrete engine type).
#pragma once
#include "AutoModel/modeling_llama3.hpp"
#include "models/llama/llama_npu_pure_xrt.hpp"


/************              llama3 (pure/native HRX)            **************/
class Llama3PureXrt : public Llama3 {
public:
    Llama3PureXrt(flm_rt::device* npu_device_inst);

    void load_model(std::string model_path, json model_inf, int default_context_length = -1, bool enable_preemption = false) override;
};
