/// \file qwen35_rai.hpp
/// \brief Qwen3.5 4B and 9B text on ryzenai-corelib, from a Q8_0 GGUF
#pragma once

#include "causal_lm.hpp"
#include "lm_config.hpp"
#include "models/qwen35/rai/aie_next/qwen35_rai_gguf.hpp"
#include "rai/corelib_runtime.hpp"

#include <cstdint>
#include <memory>

namespace flm::qwen35 {

/// \brief dense Qwen3.5 on the NPU through ryzenai-corelib
/// \note Vision and MTP are not this engine. Prefill steps the decode chain,
///       which is the reference the batched prefill path is checked against.
class qwen35_rai final : public causal_lm {
public:
    qwen35_rai(LM_Config config, std::shared_ptr<Qwen35Gguf> package,
               std::shared_ptr<corelib::CorelibRuntime> runtime,
               std::uint32_t max_length = kMaxSequenceLength);
    ~qwen35_rai() override;

    buffer<bf16> forward(int id) override;
    buffer<bf16> prefill(std::vector<int>& ids, void* payload = nullptr) override;
    void set_context_length(int length) override;
    void load_weights(Q4NX&) override;
    void update_max_length(std::uint32_t max_length) override;
    void clear_context() override;
    buffer<bf16> get_k_cache(int layer, int index) override;
    buffer<bf16> get_v_cache(int layer, int index) override;
    int get_current_context_length() override;
    int checkpoint() override;
    int restore() override;
    bool poisoned() const noexcept;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace flm::qwen35
