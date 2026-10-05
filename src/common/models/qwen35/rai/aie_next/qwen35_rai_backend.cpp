#include "models/qwen35/rai/aie_next/qwen35_rai_backend.hpp"

#include "models/qwen35/rai/aie_next/qwen35_rai.hpp"
#include "rai/corelib_runtime.hpp"
#include "utils/utils.hpp"

#include <filesystem>
#include <memory>
#include <stdexcept>
#include <string>

namespace flm::qwen35 {
namespace {

std::string GgufName(const nlohmann::ordered_json& model_info) {
    std::string found;
    if (model_info.contains("files") && model_info["files"].is_array()) {
        for (const auto& file : model_info["files"]) {
            if (!file.is_string()) continue;
            const auto name = file.get<std::string>();
            if (name.size() > 5 && name.ends_with(".gguf")) {
                if (!found.empty())
                    throw std::runtime_error("Qwen3.5 rai entry lists more than one .gguf file");
                found = name;
            }
        }
    }
    if (found.empty()) throw std::runtime_error("Qwen3.5 rai entry lists no .gguf file");
    return found;
}

class RaiBackend final : public flm::backend::ModelBackend {
public:
    explicit RaiBackend(const flm::backend::BackendContext& context) {
        if (!context.config) throw std::runtime_error("Qwen3.5 rai backend needs an LM_Config");
        if (context.enable_preemption)
            throw std::invalid_argument("Qwen3.5 rai does not support preemption");
        if (context.context_length < 1 || context.context_length > kRaiContextLimit)
            throw std::out_of_range("Qwen3.5 rai context length must be in 1..4096");

        auto package = Qwen35Gguf::Open(
            std::filesystem::path(context.model_path) / GgufName(context.model_info));
        runtime_ = corelib::CorelibRuntime::GetOrCreate(utils::get_executable_directory());
        auto engine = std::make_unique<qwen35_rai>(*context.config, std::move(package), runtime_,
                                                   context.context_length);
        engine->clear_context();
        engine_ = std::move(engine);
    }

    ~RaiBackend() override { engine_.reset(); }

    causal_lm& engine() override { return *engine_; }
    std::string id() const override { return flm::backend::kRaiBackendId; }
    std::string detail() const override {
        return runtime_ ? runtime_->loaded_library_path().string() : std::string();
    }
    std::uint32_t max_decode_length() const override { return kRaiDecodeLimit; }
    bool supports_preemption() const override { return false; }
    bool forwards_past_eos() const override { return false; }
    bool poisoned() const noexcept override { return engine_ && engine_->poisoned(); }
    std::optional<std::vector<int>> forced_eos_ids() const override {
        return std::vector<int>(kEosIds.begin(), kEosIds.end());
    }

private:
    std::shared_ptr<corelib::CorelibRuntime> runtime_;
    std::unique_ptr<qwen35_rai> engine_;
};

}  // namespace

flm::backend::BackendFactory rai_factory() {
    return [](const flm::backend::BackendContext& context)
               -> std::unique_ptr<flm::backend::ModelBackend> {
        return std::make_unique<RaiBackend>(context);
    };
}

}  // namespace flm::qwen35
