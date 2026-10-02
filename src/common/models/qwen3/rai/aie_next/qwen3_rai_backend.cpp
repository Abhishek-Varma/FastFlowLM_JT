/// \file qwen3_rai_backend.cpp
/// \brief The ryzenai-corelib backend for Qwen3
#include "models/qwen3/rai/aie_next/qwen3_rai_backend.hpp"

#include "models/qwen3/rai/aie_next/qwen3_rai.hpp"
#include "models/qwen3/rai/aie_next/qwen3_rai_gguf.hpp"
#include "rai/corelib_runtime.hpp"
#include "utils/file_access.hpp"
#include "utils/utils.hpp"

#include <filesystem>
#include <fstream>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace flm::qwen3 {
namespace {

nlohmann::json ReadJson(const std::filesystem::path& path) {
    flm::file_access::ObserveOpen(path);
    std::ifstream input(path, std::ios::binary);
    if (!input) throw std::runtime_error("Cannot open " + path.string());
    try {
        return nlohmann::json::parse(input);
    } catch (const std::exception& error) {
        throw std::runtime_error("Cannot parse " + path.string() + ": " + error.what());
    }
}

/// \brief the one .gguf the catalog entry lists
/// \note One backend serves four sizes, so the file name comes from the entry
///       rather than from a constant.
std::string GgufName(const nlohmann::ordered_json& model_info) {
    std::string found;
    if (model_info.contains("files") && model_info["files"].is_array()) {
        for (const auto& file : model_info["files"]) {
            if (!file.is_string()) continue;
            const auto name = file.get<std::string>();
            if (name.size() > 5 && name.ends_with(".gguf")) {
                if (!found.empty())
                    throw std::runtime_error("Qwen3 rai entry lists more than one .gguf file");
                found = name;
            }
        }
    }
    if (found.empty()) throw std::runtime_error("Qwen3 rai entry lists no .gguf file");
    return found;
}

class RaiBackend final : public flm::backend::ModelBackend {
public:
    explicit RaiBackend(const flm::backend::BackendContext& context) {
        if (!context.config) throw std::runtime_error("Qwen3 rai backend needs an LM_Config");
        if (context.enable_preemption)
            throw std::invalid_argument("Qwen3 rai does not support preemption");
        if (context.context_length < 1 || context.context_length > kRaiContextLimit)
            throw std::out_of_range("Qwen3 rai context length must be in 1..4096");

        // Validate every source of truth before the runtime or any device
        // object exists, so a mismatched package fails with nothing allocated.
        const std::filesystem::path root(context.model_path);
        const auto config = ReadJson(root / "config.json");
        const auto tokenizer = ReadJson(root / "tokenizer.json");
        const auto tokenizer_config = context.tokenizer_config
                                          ? *context.tokenizer_config
                                          : ReadJson(root / "tokenizer_config.json");
        auto package = Qwen3GgufPackage::Open(root / GgufName(context.model_info));
        package->ValidateContract(config, tokenizer, tokenizer_config);

        runtime_ = corelib::CorelibRuntime::GetOrCreate(utils::get_executable_directory());
        auto engine = std::make_unique<qwen3_rai>(*context.config, std::move(package), runtime_,
                                                  context.context_length);
        engine->clear_context();
        engine_ = std::move(engine);
    }

    ~RaiBackend() override {
        // corelib objects must not outlive the API they came from.
        engine_.reset();
    }

    causal_lm& engine() override { return *engine_; }
    std::string id() const override { return flm::backend::kRaiBackendId; }
    std::string detail() const override {
        return runtime_ ? runtime_->loaded_library_path().string() : std::string();
    }
    std::uint32_t max_decode_length() const override { return kRaiDecodeLimit; }
    bool supports_preemption() const override { return false; }
    /// \note corelib rejects a decode past its own limit, so the extra forward()
    ///       the FastFlowLM engines want after an EOS token would fail here.
    bool forwards_past_eos() const override { return false; }
    bool poisoned() const noexcept override { return engine_ && engine_->poisoned(); }
    std::optional<std::vector<int>> forced_eos_ids() const override {
        return std::vector<int>(kEosIds.begin(), kEosIds.end());
    }

private:
    // Declared first so it outlives the engine.
    std::shared_ptr<corelib::CorelibRuntime> runtime_;
    std::unique_ptr<qwen3_rai> engine_;
};

}  // namespace

flm::backend::BackendFactory rai_factory() {
    return [](const flm::backend::BackendContext& context)
               -> std::unique_ptr<flm::backend::ModelBackend> {
        return std::make_unique<RaiBackend>(context);
    };
}

}  // namespace flm::qwen3
