/// \file auto_decision_model.cpp
/// \brief AutoDecisionModel: the d3 contract around the d3_npu engine (see the header).
#include "AutoDecisionModel/auto_decision_model.hpp"

#include <cstring>
#include <filesystem>
#include <fstream>
#include <sstream>

#include "minja/chat-template.hpp"
#include "utils/utils.hpp"
#ifdef FLM_HAS_D3_NPU
#include "models/d3/flm/aie2p/d3_npu.hpp"
#endif

std::unordered_set<std::string> decisionModelTags = {"d3:lite"};

namespace {

std::string read_text(const std::filesystem::path& p) {
    std::ifstream f(p, std::ios::binary);
    if (!f) throw std::runtime_error("cannot open " + p.string());
    std::stringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

/// readout.safetensors: {"weight": F32 [255, hidden]}
std::vector<float> load_readout(const std::filesystem::path& p, int& hidden) {
    const std::string blob = read_text(p);
    if (blob.size() < 8) throw std::runtime_error("readout.safetensors is truncated");
    uint64_t n = 0;
    std::memcpy(&n, blob.data(), 8);
    if (8 + n > blob.size()) throw std::runtime_error("readout.safetensors header is truncated");
    const auto header = nlohmann::json::parse(blob.substr(8, n));
    if (!header.contains("weight")) throw std::runtime_error("readout.safetensors has no weight tensor");
    const auto& w = header["weight"];
    const auto shape = w["shape"].get<std::vector<int64_t>>();
    const auto off = w["data_offsets"].get<std::vector<uint64_t>>();
    if (w["dtype"] != "F32" || shape.size() != 2 || shape[0] != flm::d3::MAX_OPTIONS) {
        throw std::runtime_error("readout weight must be F32 [255, hidden]");
    }
    hidden = static_cast<int>(shape[1]);
    const size_t bytes = off[1] - off[0];
    if (bytes != static_cast<size_t>(shape[0] * shape[1]) * sizeof(float) || 8 + n + off[1] > blob.size()) {
        throw std::runtime_error("readout weight data does not match its shape");
    }
    std::vector<float> out(static_cast<size_t>(shape[0] * shape[1]));
    std::memcpy(out.data(), blob.data() + 8 + n + off[0], bytes);
    return out;
}

}  // namespace

AutoDecisionModel::AutoDecisionModel(flm_rt::device* npu_device_inst, std::string model_tag)
    : npu_device_inst_(npu_device_inst), tag_(std::move(model_tag)) {}

AutoDecisionModel::~AutoDecisionModel() = default;

std::vector<int> AutoDecisionModel::encode(const std::string& text) {
    return tokenizer_->encode(text);
}

void AutoDecisionModel::load_model(const std::string& model_path, const nlohmann::json& model_info,
                                   bool enable_preemption) {
    (void)model_info;
    namespace fs = std::filesystem;
    const fs::path root(model_path);
    header_print("FLM", "Loading decision model: " << model_path);

    // decision_config.json: the same checks as d3_runtime.D3.__init__
    const auto dc = nlohmann::json::parse(read_text(root / "decision_config.json"));
    if (dc.value("format_version", 0) != 1) throw std::runtime_error("decision_config.json: format_version must be 1");
    if (dc.value("format_id", std::string()) != flm::d3::FORMAT_ID) throw std::runtime_error("decision_config.json: unsupported format_id");
    if (dc.value("prompt", std::string("d3")) != "d3") throw std::runtime_error("decision_config.json: prompt must be d3");
    if (dc.value("attention_mode", std::string("causal")) != "noncausal_full_attention") {
        throw std::runtime_error("decision_config.json: this engine runs noncausal_full_attention checkpoints only");
    }
    if (dc.value("pooling", std::string("last")) != "last") throw std::runtime_error("decision_config.json: pooling must be last");
    if (dc.value("readout_dtype", std::string("float32")) != "float32") throw std::runtime_error("decision_config.json: readout_dtype must be float32");
    temperature_ = dc.value("temperature", 1.0f);
    if (!(std::isfinite(temperature_) && temperature_ > 0)) throw std::runtime_error("decision_config.json: temperature must be finite and positive");
    const auto codes = dc["codes"].get<std::vector<std::string>>();
    const auto token_ids = dc["token_ids"].get<std::vector<int>>();

    // the tokenizer must give exactly the package's answer codes (upstream refuses otherwise)
    tokenizer_ = std::make_unique<Tokenizer>(model_path);
    std::vector<std::string> got_codes;
    std::vector<int> got_ids;
    flm::d3::answer_codes([this](const std::string& s) { return encode(s); }, got_codes, got_ids);
    if (got_codes != codes || got_ids != token_ids) {
        throw std::runtime_error("the answer codes or token ids in decision_config.json differ from the ones this tokenizer gives");
    }
    if (encode("State:\nA") != encode("State:\nA")) throw std::runtime_error("tokenizer is not deterministic");
    codes_ = codes;

    // the package's chat template must render the d3 messages exactly as the contract does
    {
        std::string tmpl;
        if (fs::exists(root / "chat_template.jinja")) {
            tmpl = read_text(root / "chat_template.jinja");
        } else {
            tmpl = nlohmann::json::parse(read_text(root / "tokenizer_config.json")).value("chat_template", std::string());
        }
        minja::chat_template chat(tmpl, "", "");
        minja::chat_template_inputs in;
        in.add_generation_prompt = true;
        in.extra_context = nlohmann::ordered_json{{"enable_thinking", false}};
        const std::string user = "State:\n(empty)\n\nQuestion:\nprobe\n\nOptions:\nA: x\n\nReply with only the code of the best option.";
        in.messages = nlohmann::ordered_json::array({
            {{"role", "system"}, {"content", flm::d3::SYSTEM_PROMPT}},
            {{"role", "user"}, {"content", user}},
        });
        if (chat.apply(in) != flm::d3::chat_text(flm::d3::SYSTEM_PROMPT, user)) {
            throw std::runtime_error("chat_template.jinja renders the d3 prompt differently from the d3 contract");
        }
    }

    readout_ = load_readout(root / "readout.safetensors", hidden_);

    // the engine
    config_ = std::make_unique<LM_Config>();
    config_->from_pretrained(model_path);
#ifdef FLM_HAS_D3_NPU
    npu_ = std::make_unique<npu_xclbin_manager>(npu_device::device_npu2, npu_device_inst_, enable_preemption);
    auto engine = std::make_unique<d3_npu>(*config_, npu_.get(), MAX_INPUT_TOKENS);
    {
        Q4NX q4nx(model_path);
        engine->load_weights(q4nx);
    }
    if (engine->hidden_size() != hidden_) throw std::runtime_error("readout width differs from the model's hidden size");
    engine_ = std::move(engine);
#else
    (void)enable_preemption;
    throw std::runtime_error("this FastFlowLM build has no d3 engine (libd3_npu)");
#endif

    // replies name the upstream model, as d3_server does ("model": "d3-lite")
    const auto& marker = config_->sub("decision");
    name_ = marker.is_object() ? marker.value("model_name", std::string("d3-lite")) : std::string("d3-lite");
    header_print("FLM", "Decision model ready: " << name_ << " (" << tag_ << "), max input " << MAX_INPUT_TOKENS << " tokens");
}

std::map<std::string, std::vector<double>> AutoDecisionModel::run(const flm::d3::Prepared& prepared) {
    const auto keys = prepared.runnable();
    std::map<std::string, std::vector<double>> out;
    if (keys.empty()) return out;
    std::vector<std::vector<int>> ids;
    ids.reserve(keys.size());
    for (const auto& k : keys) ids.push_back(prepared.sequences.at(k));
    // moved, never copied: XRT buffer copies do not own their memory
    std::vector<buffer<bf16>> hidden = engine_->encode_batch(ids);
    std::vector<float> h(hidden_);
    for (size_t i = 0; i < keys.size(); i++) {
        const bf16* row = hidden[i].data();
        for (int d = 0; d < hidden_; d++) h[d] = static_cast<float>(row[d]);
        const int count = static_cast<int>(prepared.questions.at(keys[i]).keys.size());
        auto p = flm::d3::readout_probabilities(h.data(), readout_.data(), hidden_, count, temperature_);
        out[keys[i]] = std::vector<double>(p.begin(), p.end());
    }
    return out;
}

DecisionReply AutoDecisionModel::system_one(const std::string& raw_body) {
    DecisionReply reply;
    flm::d3::PyValue body;
    try {
        body = flm::d3::parse_json(raw_body);
    } catch (const flm::d3::JsonError& e) {
        reply.status = 422;
        reply.body = {{"detail", std::string("could not read the body as JSON (") + e.what() + ")"}};
        return reply;
    }
    flm::d3::Prepared prepared;
    try {
        prepared = flm::d3::prepare(body, codes_);
    } catch (const flm::d3::RequestError& e) {
        reply.status = 422;
        reply.body = {{"detail", e.detail}};
        return reply;
    }
    flm::d3::tokenize(prepared, [this](const std::string& s) { return encode(s); }, MAX_INPUT_TOKENS);
    const auto invalid = flm::d3::invalid_questions(prepared);
    if (!invalid.empty()) {
        reply.status = 422;
        reply.body = {{"detail", {{"invalid_questions", invalid}}}};
        return reply;
    }
    std::lock_guard<std::mutex> lock(turn_);
    const auto probabilities = run(prepared);
    reply.body = flm::d3::respond(prepared, probabilities, name_);
    return reply;
}

void AutoDecisionModel::warmup() {
    // one pass per GEMM row bucket a typical request lands in (512 and 1024 rows); the engine keeps
    // its sequences and buffers per bucket, so these are what the first requests would pay for
    std::lock_guard<std::mutex> lock(turn_);
    for (int len : {330, 700}) {
        auto warm = engine_->encode_batch({std::vector<int>(len, 32)});
        (void)warm;
    }
}
