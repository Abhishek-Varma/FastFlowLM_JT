/// \file auto_decision_model.cpp
/// \brief AutoDecisionModel: the d3 contract around the d3_npu engine (see the header).
#include "AutoDecisionModel/auto_decision_model.hpp"

#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <sstream>

#include "image_process_utils/imageproc.hpp"
#include "image_process_utils/imageprocAVX512.hpp"
#include "minja/chat-template.hpp"
#include "models/d3/flm/aie2p/d3_npu.hpp"   // d3_image_payload (the d3_npu class only with FLM_HAS_D3_NPU)
#include "utils/utils.hpp"

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

// The Qwen2-VL image processor as d3-lite's processor configures it (preprocessor_config.json): RGB
// resized (bicubic, antialiased) to the grid smart_resize chose, rescaled by 1/255, normalized with
// mean = std = 0.5, the frame repeated for the 2-frame temporal patch, and cut into 16 x 16 patches in
// 2 x 2 merge order.
constexpr int TEMPORAL_PATCH = 2;
constexpr float RESCALE = 1.0f / 255.0f;
constexpr float IMAGE_MEAN = 0.5f;
constexpr float IMAGE_STD = 0.5f;

/// Append one decoded image's patches (RGB HWC, already decoded) to `out`, resized to grid_h x grid_w patches.
bool preprocess_image(ImageReader& reader, image_data_t& rgb, int grid_h, int grid_w, std::vector<bf16>& out) {
    image_data_t chw;
    if (!reader.reorder_hwc_to_chw(rgb, chw)) return false;
    const int w = grid_w * flm::d3::IMAGE_PATCH_SIZE, h = grid_h * flm::d3::IMAGE_PATCH_SIZE;
    std::vector<uint8_t> resized = imgproc::avx512::resize_bicubic_antialias_rgb_planar_avx512(
        chw.pixels.data(), chw.width, chw.height, w, h, true);
    reader.recycle(chw);
    const size_t frame = static_cast<size_t>(w) * h * 3;
    std::vector<float> scratch(frame * TEMPORAL_PATCH);
    imgproc::avx512::rescale_and_normalize_avx512(resized.data(), scratch.data(), w, h, 3, true, RESCALE,
                                                 true, IMAGE_MEAN, IMAGE_STD);
    for (int t = 1; t < TEMPORAL_PATCH; t++) std::memcpy(scratch.data() + t * frame, scratch.data(), frame * sizeof(float));
    const size_t at = out.size();
    out.resize(at + frame * TEMPORAL_PATCH);
    imgproc::reorder_patches_inplace(scratch.data(), out.data() + at, 1, 1, TEMPORAL_PATCH, 3, grid_h, grid_w,
                                     flm::d3::IMAGE_MERGE_SIZE, flm::d3::IMAGE_PATCH_SIZE);
    return true;
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
        // and the image prompt (image_messages: the images, then the text)
        in.messages = nlohmann::ordered_json::array({
            {{"role", "system"}, {"content", flm::d3::SYSTEM_PROMPT}},
            {{"role", "user"}, {"content", nlohmann::ordered_json::array({
                {{"type", "image"}}, {{"type", "image"}}, {{"type", "text"}, {"text", user}}})}},
        });
        const std::string pair = std::string("<|vision_start|>") + flm::d3::IMAGE_TOKEN + "<|vision_end|>";
        if (chat.apply(in) != flm::d3::chat_text(flm::d3::SYSTEM_PROMPT, pair + pair + user)) {
            throw std::runtime_error("chat_template.jinja renders the d3 image prompt differently from the d3 contract");
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

std::map<std::string, std::vector<double>> AutoDecisionModel::run(const flm::d3::Prepared& prepared, void* payload) {
    const auto keys = prepared.runnable();
    std::map<std::string, std::vector<double>> out;
    if (keys.empty()) return out;
    std::vector<std::vector<int>> ids;
    ids.reserve(keys.size());
    for (const auto& k : keys) ids.push_back(prepared.sequences.at(k));
    // moved, never copied: XRT buffer copies do not own their memory
    std::vector<buffer<bf16>> hidden = engine_->encode_batch(ids, payload);
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
    // each image is decoded right after its header passes (upstream loads them one by one)
    std::vector<image_data_t> decoded;
    const auto decode = [&](size_t, const flm::d3::ImageFile& f) {
        image_data_t rgb;
        if (!image_reader_.load_image_bytes(f.data, rgb)) {
            throw flm::d3::ContractError("the image could not be decoded (the file is truncated or corrupt)");
        }
        if (rgb.width != f.width || rgb.height != f.height) {
            image_reader_.recycle(rgb);
            throw flm::d3::ContractError("the image could not be decoded (its pixels do not match its header)");
        }
        decoded.push_back(std::move(rgb));
    };
    const auto release = [&]() {
        for (auto& d : decoded) image_reader_.recycle(d);
        decoded.clear();
    };
    try {
        prepared = flm::d3::prepare(body, codes_, decode);
    } catch (const flm::d3::RequestError& e) {
        release();
        reply.status = 422;
        reply.body = {{"detail", e.detail}};
        return reply;
    }
    flm::d3::tokenize(prepared, [this](const std::string& s) { return encode(s); }, MAX_INPUT_TOKENS);
    const auto invalid = flm::d3::invalid_questions(prepared);
    if (!invalid.empty()) {
        release();
        reply.status = 422;
        reply.body = {{"detail", {{"invalid_questions", invalid}}}};
        return reply;
    }
    // the processor's pixels, once per request (every question shares the images)
    d3_image_payload payload;
    for (size_t i = 0; i < decoded.size(); i++) {
        const auto [gh, gw] = prepared.image_grids[i];
        if (!preprocess_image(image_reader_, decoded[i], gh, gw, payload.pixels)) {
            release();
            throw std::runtime_error("could not prepare image " + std::to_string(i));
        }
        payload.images.push_back({gh, gw});
    }
    release();
    if (!payload.images.empty()) {
        const std::vector<int> pad = encode(flm::d3::IMAGE_TOKEN);
        payload.image_token_id = pad.at(0);
        if (const char* dir = std::getenv("D3_DUMP_PIXELS")) {
            // rig: the processor's output, to compare with the upstream processor's pixel_values
            const std::string stem = std::string(dir) + "/request_" + std::to_string(dump_count_++);
            std::ofstream(stem + ".bin", std::ios::binary)
                .write(reinterpret_cast<const char*>(payload.pixels.data()), payload.pixels.size() * sizeof(bf16));
            nlohmann::json grids = nlohmann::json::array();
            for (const auto& im : payload.images) grids.push_back({1, im.grid_h, im.grid_w});
            std::ofstream(stem + ".json") << grids.dump();
        }
    }
    std::lock_guard<std::mutex> lock(turn_);
    const auto probabilities = run(prepared, payload.images.empty() ? nullptr : &payload);
    reply.body = flm::d3::respond(prepared, probabilities, name_);
    return reply;
}

void AutoDecisionModel::warmup() {
    // One pass per GEMM row bucket typical requests land in: 2,048 rows (several questions packed),
    // 1,024 and 512 (one question). The engine keeps each bucket's instruction sequences for its
    // lifetime and the scratch of buckets that fit one 4,096-row pass together (these three do), so
    // the first requests in each bucket run at steady-state speed.
    std::lock_guard<std::mutex> lock(turn_);
    const std::vector<std::vector<std::vector<int>>> passes = {
        std::vector<std::vector<int>>(4, std::vector<int>(500, 32)),   // 2,000 rows -> 2,048
        {std::vector<int>(700, 32)},                                   // 1,024
        {std::vector<int>(330, 32)},                                   // 512
    };
    for (const auto& ids : passes) {
        auto warm = engine_->encode_batch(ids);
        (void)warm;
    }
}
