/// \file modeling_qwen3_pure_xrt.hpp
/// \brief Qwen3PureXrt wrapper: identical behaviour to Qwen3 but backed by the
///        qwen3_npu_pure_xrt engine, whose hot dispatch paths call the native
///        HRX C API directly instead of the flm_rt (hrx::run/runlist) shim.
/// \note  Only load_model (engine instantiation) and insert (concrete-type
///        checkpoint/restore) differ from Qwen3; generation and parsing are
///        inherited unchanged.
#pragma once
#include "AutoModel/modeling_qwen3.hpp"
#include "models/qwen3/qwen3_npu_pure_xrt.hpp"


/************              qwen3 (pure/native HRX)            **************/
class Qwen3PureXrt : public Qwen3 {
public:
    Qwen3PureXrt(flm_rt::device* npu_device_inst);

    void load_model(std::string model_path, json model_inf, int default_context_length = -1, bool enable_preemption = false) override;
    bool insert(chat_meta_info_t& meta_info, lm_uniform_input_t& input, std::function<bool()> is_cancelled = [] { return false; }) override;
};
