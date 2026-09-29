/// \file qwen3_rai_gguf.hpp
/// \brief A Qwen3 Q8_0 GGUF, validated and resolved before any device work
#pragma once

#include "models/qwen3/rai/aie_next/qwen3_rai_config.hpp"
#include "rai/gguf_file.hpp"

#include <filesystem>
#include <memory>
#include <vector>

#include <nlohmann/json.hpp>

namespace flm::qwen3 {

/// \brief one decoder block's tensors, as spans over the mapping
/// \note Nothing is fused in a Qwen3 conversion: Q, K and V are three tensors
///       and gate and up two, so each view is a whole tensor.
struct Qwen3LayerTensors {
    rai::GgufFloatTensor attn_norm, ffn_norm, q_norm, k_norm;
    rai::GgufTensor q, k, v, o, gate, up, down;
};

class Qwen3GgufPackage final {
public:
    /// \brief map the file, select its size and shape-check every tensor
    /// \throws std::runtime_error naming the field at the first mismatch
    static std::shared_ptr<Qwen3GgufPackage> Open(const std::filesystem::path& gguf_path);
    ~Qwen3GgufPackage();

    const Qwen3Config& Config() const noexcept;
    const std::filesystem::path& Path() const;
    const rai::GgufTensor& Embedding() const noexcept;
    /// \brief the tensor lm_head packs from
    /// \note A property of the file, not the family: Qwen3-8B ships its own
    ///       output.weight, the smaller three tie lm_head to token_embd. The
    ///       two are the same shape, so reading the wrong one raises nothing.
    const rai::GgufTensor& LmHead() const noexcept;
    bool TiedLmHead() const noexcept;
    const rai::GgufFloatTensor& OutputNorm() const noexcept;
    const Qwen3LayerTensors& Layer(std::size_t layer) const;

    /// \brief cross-check the file against the package's JSON files
    /// \note config.json is checked only for the keys it states: FastFlowLM's
    ///       Qwen3-4B config omits tie_word_embeddings.
    void ValidateContract(const nlohmann::json& config,
                          const nlohmann::json& tokenizer,
                          const nlohmann::json& tokenizer_config) const;

private:
    Qwen3GgufPackage() = default;

    std::shared_ptr<rai::GgufFile> file_;
    const Qwen3Config* config_{};
    rai::GgufTensor embedding_{}, lm_head_{};
    rai::GgufFloatTensor output_norm_{};
    std::vector<Qwen3LayerTensors> layers_;
};

}  // namespace flm::qwen3
