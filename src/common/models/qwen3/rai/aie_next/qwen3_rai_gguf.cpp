#include "models/qwen3/rai/aie_next/qwen3_rai_gguf.hpp"

#include <array>
#include <cmath>
#include <map>
#include <stdexcept>
#include <string>
#include <utility>

namespace flm::qwen3 {
namespace {
using rai::FailField;

std::string Name(std::size_t layer, const char* suffix) {
    return "blk." + std::to_string(layer) + suffix;
}

bool IsRetained(std::string_view key) {
    return key == "general.architecture" || key.starts_with("qwen3.") ||
           key == "tokenizer.ggml.eos_token_id";
}

std::int64_t SignedMetadata(const rai::GgufFile& file, std::string_view key) {
    return static_cast<std::int64_t>(file.Unsigned(key));
}

const nlohmann::json* Find(const nlohmann::json& object, std::string_view key) {
    const auto it = object.find(std::string(key));
    return it == object.end() ? nullptr : &*it;
}

void RequireInteger(const nlohmann::json& object, std::string_view key, std::int64_t expected,
                    bool optional = false) {
    const auto* value = Find(object, key);
    if (!value) {
        if (optional) return;
        FailField(key, "missing", std::to_string(expected));
    }
    if (!value->is_number_integer()) FailField(key, value->dump(), std::to_string(expected));
    if (value->get<std::int64_t>() != expected)
        FailField(key, value->dump(), std::to_string(expected));
}

void RequireClose(const nlohmann::json& object, std::string_view key, double expected) {
    const auto* value = Find(object, key);
    if (!value || !value->is_number())
        FailField(key, value ? value->dump() : "missing", std::to_string(expected));
    const double actual = value->get<double>();
    if (!std::isfinite(actual) || std::fabs(actual - expected) > 1e-6 * std::fabs(expected))
        FailField(key, value->dump(), std::to_string(expected));
}

void RequireString(const nlohmann::json& object, std::string_view key, std::string_view expected) {
    const auto* value = Find(object, key);
    if (!value || !value->is_string() || value->get_ref<const std::string&>() != expected)
        FailField(key, value ? value->dump() : "missing", std::string(expected));
}
}  // namespace

Qwen3GgufPackage::~Qwen3GgufPackage() = default;

std::shared_ptr<Qwen3GgufPackage> Qwen3GgufPackage::Open(const std::filesystem::path& gguf_path) {
    std::shared_ptr<Qwen3GgufPackage> package(new Qwen3GgufPackage());
    package->file_ = rai::GgufFile::Open(gguf_path, IsRetained);
    const auto& file = *package->file_;

    // The architecture string, not a missing tensor, is what says QK-Norm is
    // there: a Qwen2 conversion simply has no attn_q_norm to find.
    const auto architecture = file.String("general.architecture");
    if (architecture != "qwen3") FailField("general.architecture", architecture, "qwen3");
    Qwen3Shape shape{};
    shape.layers = SignedMetadata(file, "qwen3.block_count");
    shape.hidden = SignedMetadata(file, "qwen3.embedding_length");
    shape.q_heads = SignedMetadata(file, "qwen3.attention.head_count");
    shape.kv_heads = SignedMetadata(file, "qwen3.attention.head_count_kv");
    // Required, not derived: 0.6B is 16 heads of 128 over a hidden size of
    // 1024, where hidden / heads would give 64.
    shape.head_dim = SignedMetadata(file, "qwen3.attention.key_length");
    if (file.HasMetadata("qwen3.attention.value_length") &&
        SignedMetadata(file, "qwen3.attention.value_length") != shape.head_dim)
        FailField("qwen3.attention.value_length",
                  std::to_string(file.Unsigned("qwen3.attention.value_length")),
                  std::to_string(shape.head_dim));
    shape.intermediate = SignedMetadata(file, "qwen3.feed_forward_length");
    shape.rope_theta = file.Number("qwen3.rope.freq_base");
    shape.epsilon = file.Number("qwen3.attention.layer_norm_rms_epsilon");
    package->config_ = &SelectQwen3Row(shape);
    const auto& c = *package->config_;

    if (file.Unsigned("tokenizer.ggml.eos_token_id") != static_cast<std::uint64_t>(kEosIds[0]))
        FailField("tokenizer.ggml.eos_token_id",
                  std::to_string(file.Unsigned("tokenizer.ggml.eos_token_id")),
                  std::to_string(kEosIds[0]));

    const std::array<std::int64_t, 1> hidden{c.hidden};
    const std::array<std::int64_t, 1> head{c.head_dim};
    const std::array<std::int64_t, 2> table{c.vocab, c.hidden};
    package->embedding_ = file.RequireQ8("token_embd.weight", table);
    package->lm_head_ = file.HasTensor("output.weight") ? file.RequireQ8("output.weight", table)
                                                        : package->embedding_;
    package->output_norm_ = file.RequireF32("output_norm.weight", hidden);
    package->layers_.reserve(static_cast<std::size_t>(c.layers));
    for (std::size_t i = 0; i < static_cast<std::size_t>(c.layers); ++i) {
        Qwen3LayerTensors t;
        t.attn_norm = file.RequireF32(Name(i, ".attn_norm.weight"), hidden);
        t.ffn_norm = file.RequireF32(Name(i, ".ffn_norm.weight"), hidden);
        t.q_norm = file.RequireF32(Name(i, ".attn_q_norm.weight"), head);
        t.k_norm = file.RequireF32(Name(i, ".attn_k_norm.weight"), head);
        // [out, in]: Q is hidden -> q_heads * head_dim, which is NOT hidden on
        // 0.6B (2048 vs 1024) or 4B (4096 vs 2560); O is the reverse.
        t.q = file.RequireQ8(Name(i, ".attn_q.weight"), std::array<std::int64_t, 2>{c.q_dim(), c.hidden});
        t.k = file.RequireQ8(Name(i, ".attn_k.weight"), std::array<std::int64_t, 2>{c.kv_dim(), c.hidden});
        t.v = file.RequireQ8(Name(i, ".attn_v.weight"), std::array<std::int64_t, 2>{c.kv_dim(), c.hidden});
        t.o = file.RequireQ8(Name(i, ".attn_output.weight"), std::array<std::int64_t, 2>{c.hidden, c.q_dim()});
        t.gate = file.RequireQ8(Name(i, ".ffn_gate.weight"), std::array<std::int64_t, 2>{c.intermediate, c.hidden});
        t.up = file.RequireQ8(Name(i, ".ffn_up.weight"), std::array<std::int64_t, 2>{c.intermediate, c.hidden});
        t.down = file.RequireQ8(Name(i, ".ffn_down.weight"), std::array<std::int64_t, 2>{c.hidden, c.intermediate});
        package->layers_.push_back(std::move(t));
    }
    return package;
}

const Qwen3Config& Qwen3GgufPackage::Config() const noexcept { return *config_; }
const std::filesystem::path& Qwen3GgufPackage::Path() const { return file_->Path(); }
const rai::GgufTensor& Qwen3GgufPackage::Embedding() const noexcept { return embedding_; }
const rai::GgufTensor& Qwen3GgufPackage::LmHead() const noexcept { return lm_head_; }
bool Qwen3GgufPackage::TiedLmHead() const noexcept { return lm_head_.name == embedding_.name; }
const rai::GgufFloatTensor& Qwen3GgufPackage::OutputNorm() const noexcept { return output_norm_; }

const Qwen3LayerTensors& Qwen3GgufPackage::Layer(std::size_t layer) const {
    if (layer >= layers_.size())
        FailField("Qwen3 layer", std::to_string(layer), "0.." + std::to_string(layers_.size() - 1));
    return layers_[layer];
}

void Qwen3GgufPackage::ValidateContract(const nlohmann::json& config,
                                        const nlohmann::json& tokenizer,
                                        const nlohmann::json& tokenizer_config) const {
    const auto& c = *config_;
    RequireString(config, "model_type", "qwen3");
    RequireInteger(config, "num_hidden_layers", c.layers);
    RequireInteger(config, "hidden_size", c.hidden);
    RequireInteger(config, "intermediate_size", c.intermediate);
    RequireInteger(config, "num_attention_heads", c.q_heads);
    RequireInteger(config, "num_key_value_heads", c.kv_heads);
    RequireInteger(config, "head_dim", c.head_dim, true);
    RequireInteger(config, "vocab_size", c.vocab);
    RequireClose(config, "rms_norm_eps", c.epsilon);
    RequireClose(config, "rope_theta", c.rope_theta);
    if (const auto* tied = Find(config, "tie_word_embeddings")) {
        if (!tied->is_boolean() || tied->get<bool>() != TiedLmHead())
            FailField("tie_word_embeddings", tied->dump(),
                      TiedLmHead() ? "true (the GGUF has no output.weight)"
                                   : "false (the GGUF ships output.weight)");
    }

    std::map<std::string, std::int64_t, std::less<>> special;
    try {
        for (const auto& item : tokenizer.at("added_tokens"))
            special.emplace(item.at("content").get<std::string>(), item.at("id").get<std::int64_t>());
    } catch (const nlohmann::json::exception& error) {
        FailField("tokenizer.json added_tokens", error.what(), "an array of {content, id}");
    }
    for (const auto& [token, id] : std::array<std::pair<std::string_view, std::int64_t>, 5>{{
             {"<|endoftext|>", kEosIds[1]}, {"<|im_start|>", 151644}, {"<|im_end|>", kEosIds[0]},
             {"<think>", 151667}, {"</think>", 151668}}}) {
        const auto it = special.find(token);
        if (it == special.end()) FailField(token, "missing from tokenizer.json", std::to_string(id));
        if (it->second != id) FailField(token, std::to_string(it->second), std::to_string(id));
    }

    RequireString(tokenizer_config, "eos_token", "<|im_end|>");
    const auto* chat_template = Find(tokenizer_config, "chat_template");
    if (!chat_template || !chat_template->is_string())
        FailField("chat_template", chat_template ? chat_template->dump() : "missing",
                  "string containing the ChatML markers");
    for (const auto marker : {"<|im_start|>", "<|im_end|>"})
        if (chat_template->get_ref<const std::string&>().find(marker) == std::string::npos)
            FailField(marker, "missing from chat_template", "present in chat_template");
}

}  // namespace flm::qwen3
