#pragma once

#include "causal_lm.hpp"
#include "lm_config.hpp"
#include "models/qwen3/rai/aie_next/qwen3_rai_gguf.hpp"
#include "rai/corelib_runtime.hpp"

#include <cstdint>
#include <memory>

namespace flm::qwen3 {

/// \brief Qwen3 0.6B/1.7B/4B/8B on ryzenai-corelib, from a Q8_0 GGUF
/// \note corelib has no chunked prefill -- a multi-row pass must start at
///       position 0 -- so the engine remembers the tokens behind its KV cache,
///       and a prefill that continues a conversation re-runs them from the
///       front together with the new ones.
class qwen3_rai final : public causal_lm {
public:
    qwen3_rai(LM_Config config,
              std::shared_ptr<Qwen3GgufPackage> package,
              std::shared_ptr<corelib::CorelibRuntime> runtime,
              std::uint32_t max_length = 4096);
    ~qwen3_rai() override;

    buffer<bf16> forward(int id) override;
    buffer<bf16> prefill(std::vector<int>& ids, void* payload = nullptr) override;
    void set_context_length(int length) override;
    /// \brief unsupported; this engine's weights come from its GGUF package
    /// \note Present only because causal_lm.hpp is a frozen ABI. See the
    ///       definition and AutoModel/model_backend.hpp.
    /// \throws std::runtime_error always
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

}  // namespace flm::qwen3
