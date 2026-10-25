/// \file d3_composed.cpp
/// \brief Composed System One requests (see d3_composed.hpp): port of upstream `_system_one_composed`.
#include "decision/d3_composed.hpp"

#include <algorithm>
#include <cmath>
#include <set>

namespace flm::d3 {

namespace {

using compose::Part;

bool contains(const std::vector<std::string>& v, const std::string& s) {
    return std::find(v.begin(), v.end(), s) != v.end();
}

/// `mentions`: whether a string inside `value` (strings, list items, object values; never keys) holds a reference.
bool mentions(const PyValue& value, const std::vector<std::string>& refs) {
    if (value.is_string()) {
        return std::any_of(refs.begin(), refs.end(), [&](const std::string& r) { return value.text.find(r) != std::string::npos; });
    }
    if (value.is_object()) {
        return std::any_of(value.members.begin(), value.members.end(), [&](const auto& m) { return mentions(m.second, refs); });
    }
    if (value.is_array()) {
        return std::any_of(value.items.begin(), value.items.end(), [&](const PyValue& v) { return mentions(v, refs); });
    }
    return false;
}

ojson type_json(const PyValue* kind) { return kind ? to_ojson(*kind) : ojson(nullptr); }

/// `{"type", "error", "message"}`: the shape of a caught ValueError; `with_reasons` appends the reason.
ojson caught(const ojson& type, const std::string& error, const std::string& message, const std::string& reason = std::string()) {
    ojson e = error_answer(type, error, message);
    if (!reason.empty()) e["reason"] = reason;
    return e;
}

std::string sorted_list_repr(std::vector<std::string> names) {
    std::sort(names.begin(), names.end());
    std::string out = "[";
    for (size_t i = 0; i < names.size(); i++) {
        if (i) out += ", ";
        out += repr(PyValue::string(names[i]));
    }
    return out + "]";
}

}  // namespace

std::vector<size_t> ComposedRequest::rows_of(const std::string& key) const {
    std::vector<size_t> out;
    for (size_t i = 0; i < rows.size(); i++) {
        if (rows[i].key == key) out.push_back(i);
    }
    return out;
}

bool is_composed(const PyValue& body) {
    const PyValue* assets = body.get("assets");
    if (assets && assets->truthy()) return true;
    const PyValue* questions = body.get("questions");
    if (!questions || !questions->is_object()) return false;
    for (const auto& m : questions->members) {
        const PyValue* kind = m.second.is_object() ? m.second.get("type") : nullptr;
        if (kind && kind->is_string() && (kind->text == "box" || kind->text == "track")) return true;
    }
    return false;
}

ojson refusal(const PyValue* kind, const PyValue& question, const std::vector<std::string>& refs, const ComposedConfig& config) {
    const std::string k = kind && kind->is_string() ? kind->text : std::string();
    if ((k == "box" || k == "track") && !contains(config.question_types, k)) {
        ojson e = ojson::object();
        e["type"] = type_json(kind);
        e["error"] = "invalid_question";
        e["reason"] = k + "_unsupported";
        e["message"] = config.model_name + " does not answer " + k + " questions";
        return e;
    }
    if (!question.is_object() || refs.empty()) return nullptr;
    for (const char* place : {"instructions", "criteria"}) {
        const PyValue* value = question.get(place);
        if (!contains(config.image_placements, place) && value && mentions(*value, refs)) {
            ojson e = ojson::object();
            e["type"] = type_json(kind);
            e["error"] = "invalid_question";
            e["reason"] = "question_images_unsupported";
            e["message"] = config.model_name + " reads images in the state only, not in a question's " + place;
            return e;
        }
    }
    return nullptr;
}

namespace {

/// Everything one question's preparation reads and extends.
struct Context {
    const std::vector<std::string>& codes;
    const ComposedConfig& config;
    const ComposedHooks& hooks;
    const std::vector<std::string>& refs;
    const PyValue* assets = nullptr;
    ComposedRequest& request;
    std::map<std::string, int> asset_index;          //!< asset reference -> image table index
    std::map<std::string, std::string> asset_error;  //!< asset reference -> why it did not load
    std::map<std::string, size_t> segment_cache;
    int pad = -1;
    int pad_id() {
        if (pad < 0) {
            const std::vector<int> one = hooks.encode(IMAGE_TOKEN);
            if (one.size() != 1) throw std::runtime_error("the tokenizer does not keep <|image_pad|> as one token");
            pad = one[0];
        }
        return pad;
    }

    /// One image into the table: header checks, the host's decoding, the processor's resize. Throws ContractError with
    /// the text of upstream's `load_image` (the caller adds "images[n]: ").
    int add_image(const std::string& source, const PyValue& value) {
        ImageFile f = inspect_image(value);
        if (hooks.decode) hooks.decode(request.images.size(), f);
        ComposedImage im;
        im.source = source;
        im.width = f.width;
        im.height = f.height;
        try {
            const auto hw = smart_resize(f.height, f.width);
            im.grid_h = hw.first / IMAGE_PATCH_SIZE;
            im.grid_w = hw.second / IMAGE_PATCH_SIZE;
            im.tokens = im.grid_h * im.grid_w / (IMAGE_MERGE_SIZE * IMAGE_MERGE_SIZE);
        } catch (const ContractError& e) {
            im.reject = e.what();   // the processor refuses it when a question's row is built
        }
        request.images.push_back(std::move(im));
        return static_cast<int>(request.images.size()) - 1;
    }

    /// An asset used by one question: loaded once per request, failures remembered.
    int asset(const std::string& ref) {
        auto done = asset_index.find(ref);
        if (done != asset_index.end()) return done->second;
        auto bad = asset_error.find(ref);
        if (bad != asset_error.end()) throw ContractError(bad->second);
        try {
            const int idx = add_image(ref, *assets->get(ref));
            asset_index[ref] = idx;
            return idx;
        } catch (const ContractError& e) {
            asset_error[ref] = e.what();
            throw;
        }
    }
};

/// `_composed_question` + `_composed_row` up to the engine: the row of one image-path question.
ComposedRow compose_row(Context& cx, const std::string& key, const PyValue& question) {
    const PyValue* type = question.get("type");
    const std::string kind = type && type->is_string() ? type->text : std::string();
    ComposedRow row;
    row.key = key;
    row.type = type_json(type);
    row.kind = kind;
    if (kind == "box") {
        const PyValue* instructions = question.get("instructions");
        bool extra = false;
        for (const auto& m : question.members) extra = extra || (m.first != "type" && m.first != "instructions");
        if (extra || !instructions || !instructions->is_string()) throw ContractError("a box question takes type and instructions");
    } else {
        row.question = normalize_question(question);
        if (row.question.kind == "score" && cx.config.refuse_score) throw ContractError("score questions take no images here");
    }
    const int n_front = static_cast<int>(cx.request.front.size());
    const std::vector<Part> parts = compose::user_parts(cx.request.state, question, cx.codes, n_front, 0, cx.refs);
    const bool box = kind == "box";
    row.text = compose::render_parts(parts, box ? compose::suffix(false) : std::string());
    // the images of this question's own block, in reading order (a reference used twice is two images)
    std::vector<int> own;
    {
        int n = 0;
        for (const Part& p : parts) {
            if (p.kind != Part::Kind::Image || p.index >= 0) continue;
            try {
                own.push_back(cx.asset(p.text));
            } catch (const ContractError& e) {
                throw ContractError("images[" + std::to_string(n) + "]: " + e.what());
            }
            n++;
        }
    }
    const std::vector<int> plain = cx.hooks.encode(row.text);
    if (box) {
        row.readout = compose::readout_plan(compose::readout_offsets(plain, cx.hooks.token_bytes, row.text, false), false);
    } else {
        row.readout.offsets = {0};
        row.readout.counts = {static_cast<int>(row.question.keys.size())};
    }
    // the processor: every image (state images first, then this block's) must be accepted
    std::vector<int> in_order = cx.request.front;
    in_order.insert(in_order.end(), own.begin(), own.end());
    for (int idx : in_order) {
        if (!cx.request.images[idx].reject.empty()) throw ContractError(cx.request.images[idx].reject);
    }
    size_t pads = 0;
    for (size_t at = row.text.find(IMAGE_TOKEN); at != std::string::npos; at = row.text.find(IMAGE_TOKEN, at + 1)) pads++;
    if (pads != in_order.size() || row.text.find(VIDEO_TOKEN) != std::string::npos) {
        throw ContractError("the state or question contains a literal image or video placeholder token");
    }
    // each placeholder becomes its image's run of input tokens, in image order
    std::vector<int> expanded;
    expanded.reserve(plain.size());
    size_t next = 0;
    int pad = in_order.empty() ? -1 : cx.pad_id();
    for (int id : plain) {
        if (pad >= 0 && id == pad && next < in_order.size()) {
            expanded.insert(expanded.end(), cx.request.images[in_order[next]].tokens, pad);
            next++;
        } else {
            expanded.push_back(id);
        }
    }
    if (next != in_order.size()) throw ContractError("the state or question contains a literal image or video placeholder token");
    const int width = static_cast<int>(expanded.size());
    if (cx.config.max_length > 0 && width > cx.config.max_length) {
        throw MaxLengthExceeded("the question prompt has " + std::to_string(width) + " tokens, over the maximum context length of " +
                                std::to_string(cx.config.max_length) + " tokens; nothing was truncated");
    }
    int media = 0;
    for (int idx : cx.request.front) media += cx.request.images[idx].tokens - 1;
    row.state_len = state_block_length(row.text, cx.request.state, plain, cx.hooks.encode, cx.hooks.token_bytes, media, width,
                                       &cx.segment_cache);
    row.ids = std::move(expanded);
    row.state_images = cx.request.front;
    row.block_images = std::move(own);
    return row;
}

/// `_video_box`'s request checks while this build has no video: the shape error upstream gives, else `video_needed`.
[[noreturn]] void video_question(const PyValue& question, const std::string& kind) {
    std::vector<std::string> allowed = kind == "box" ? std::vector<std::string>{"type", "instructions", "at"}
                                                     : std::vector<std::string>{"type", "instructions", "rate"};
    bool extra = false;
    for (const auto& m : question.members) extra = extra || !contains(allowed, m.first);
    const PyValue* instructions = question.get("instructions");
    if (extra || !instructions || !instructions->is_string()) {
        throw ContractError("a " + kind + " question takes " + sorted_list_repr(allowed));
    }
    throw QuestionRefused("video_needed", "a " + kind + " question needs a video");
}

}  // namespace

ComposedRequest prepare_composed(const PyValue& body, const std::vector<std::string>& codes, const ComposedConfig& config,
                                 const ComposedHooks& hooks) {
    PrepareOptions options;
    options.unsupported_for = config.tag;
    options.composed = true;
    options.reasons = true;
    options.refuse_video_questions = config.refuse_video_questions && contains(config.question_types, "track");
    const PyValue& questions = check_request(body, options);
    const std::string tail = " by this build of " + config.tag + " (it answers choice, noul, score and box questions about text and images)";

    ComposedRequest req;
    const PyValue* assets = body.get("assets");
    const bool has_assets = assets && assets->truthy();
    if (has_assets && !assets->is_object()) {
        throw RequestError("assets takes an object of reference strings to base64 data URLs");
    }
    std::vector<std::string> refs;
    Context cx{codes, config, hooks, refs, has_assets ? assets : nullptr, req, {}, {}, {}, -1};

    // the request's images (strict, in order), then videos (not in this build)
    const PyValue* images = body.get("images");
    size_t n_images = 0;
    if (images && !images->is_null()) {
        if (!images->is_array()) throw RequestError("images takes a list of base64 data URLs");
        for (size_t i = 0; i < images->items.size(); i++) {
            try {
                req.front.push_back(cx.add_image("images[" + std::to_string(i) + "]", images->items[i]));
            } catch (const ContractError& e) {
                throw RequestError("images[" + std::to_string(i) + "]: " + e.what());
            }
        }
        n_images = images->items.size();
    }
    const PyValue* videos = body.get("videos");
    if (videos && !videos->is_null()) {
        if (!videos->is_array()) throw RequestError("videos takes a list of base64 data URLs");
        if (!videos->items.empty()) throw RequestError("videos are not supported" + tail);
    }
    for (const auto& m : questions.members) req.keys.push_back(m.first);
    const PyValue state = body.get("state") ? *body.get("state") : PyValue::null();
    try {
        check_json_value(state);
    } catch (const JsonError& e) {
        throw RequestError(std::string("not representable as JSON: ") + e.what());
    }

    // D3._system_one_composed: references, hoisting, the state's images
    if (has_assets) {
        std::vector<std::string> names;
        for (const auto& m : assets->members) names.push_back(m.first);
        try {
            refs = compose::check_references(names);
        } catch (const ContractError& e) {
            throw RequestError(e.what());
        }
    }
    compose::Hoisted hoisted = compose::hoist_state(state, refs, static_cast<int>(n_images));
    req.state = std::move(hoisted.state);
    for (size_t j = 0; j < hoisted.uses.size(); j++) {
        const size_t number = n_images + j;
        try {
            req.front.push_back(cx.asset(hoisted.uses[j]));
        } catch (const ContractError& e) {
            throw RequestError("images[" + std::to_string(number) + "]: " + e.what());
        }
    }

    for (const auto& m : questions.members) {
        const std::string& key = m.first;
        const PyValue& question = m.second;
        const PyValue* kind = question.is_object() ? question.get("type") : nullptr;
        const std::string k = kind && kind->is_string() ? kind->text : std::string();
        const ojson refused = refusal(kind, question, refs, config);
        if (!refused.is_null()) {
            req.errors[key] = refused;
            continue;
        }
        const ojson type = type_json(kind);
        try {
            if (!question.is_object()) throw ContractError("a question is an object with a type");
            const PyValue* at = question.get("at");
            if (k == "track" || (k == "box" && at && !at->is_null())) video_question(question, k);
            req.rows.push_back(compose_row(cx, key, question));
        } catch (const MaxLengthExceeded& e) {
            req.errors[key] = caught(type, "max_length_exceeded", e.what());
        } catch (const QuestionRefused& e) {
            req.errors[key] = caught(type, "invalid_question", e.what(), e.reason);
        } catch (const ContractError& e) {
            req.errors[key] = caught(type, "invalid_question", e.what());
        }
    }
    if (!req.images.empty()) req.image_token_id = cx.pad_id();
    return req;
}

std::vector<std::vector<size_t>> state_groups(const ComposedRequest& request) {
    std::vector<std::string> order;
    std::map<std::string, std::vector<size_t>> by_state;
    for (size_t i = 0; i < request.rows.size(); i++) {
        const std::string& s = request.rows[i].state_key;
        if (!by_state.count(s)) order.push_back(s);
        by_state[s].push_back(i);
    }
    std::vector<std::vector<size_t>> out;
    for (const auto& s : order) out.push_back(by_state[s]);
    return out;
}

EngineStateRequest build_engine_request(const ComposedRequest& request, const std::vector<size_t>& group,
                                        const std::function<EngineImage(size_t)>& image) {
    EngineStateRequest out;
    out.image_token_id = request.image_token_id;
    std::map<int, int> compact;   // image table index -> position in out.images
    const auto use = [&](int table_index) {
        auto it = compact.find(table_index);
        if (it != compact.end()) return it->second;
        EngineImage im = image ? image(static_cast<size_t>(table_index)) : EngineImage();
        if (im.grid_h == 0) {
            im.grid_h = request.images[table_index].grid_h;
            im.grid_w = request.images[table_index].grid_w;
        }
        out.images.push_back(std::move(im));
        compact[table_index] = static_cast<int>(out.images.size()) - 1;
        return compact[table_index];
    };
    const ComposedRow& first = request.rows.at(group.at(0));
    out.state_ids.assign(first.ids.begin(), first.ids.begin() + first.state_len);
    for (int idx : first.state_images) out.state_image_ids.push_back(use(idx));
    for (size_t r : group) {
        const ComposedRow& row = request.rows.at(r);
        // upstream (_run_state_once) refuses requests whose questions do not share one state block
        if (row.state_len != first.state_len || !std::equal(out.state_ids.begin(), out.state_ids.end(), row.ids.begin())) {
            throw std::runtime_error("the questions of a request do not share the state block's tokens");
        }
        EngineBlock block;
        block.tag = row.key;
        block.ids.assign(row.ids.begin() + row.state_len, row.ids.end());
        block.readout_from_end = row.readout.offsets;
        for (int idx : row.block_images) block.image_ids.push_back(use(idx));
        out.blocks.push_back(std::move(block));
    }
    return out;
}

std::vector<std::vector<double>> row_probabilities(const ComposedRow& row, const std::vector<float>& hidden_rows, const float* readout,
                                                   int hidden_size, float temperature) {
    std::vector<std::vector<double>> values;
    const size_t positions = row.readout.offsets.size();
    if (hidden_rows.size() != positions * static_cast<size_t>(hidden_size)) {
        throw std::runtime_error("the engine returned " + std::to_string(hidden_rows.size() / std::max(hidden_size, 1)) +
                                 " hidden rows for " + std::to_string(positions) + " readout positions");
    }
    for (size_t p = 0; p < positions; p++) {
        const auto probs = readout_probabilities(hidden_rows.data() + p * hidden_size, readout, hidden_size, row.readout.counts[p],
                                                 temperature);
        values.emplace_back(probs.begin(), probs.end());
    }
    return values;
}

ojson row_answer(const ComposedRow& row, const std::vector<std::vector<double>>& values, const ComposedConfig& config) {
    if (row.kind == "box") {
        const std::vector<std::vector<double>> edges(values.end() - 4, values.end());
        const compose::Box box = compose::decode(edges);
        return compose::box_answer(box, compose::calibrate(compose::confidence(edges, box), config.box_calibration));
    }
    return product_answer(row.question, values.at(0));
}

ojson respond_composed(const ComposedRequest& request, const std::vector<std::vector<std::vector<double>>>& values,
                       const ComposedConfig& config) {
    ojson answers = ojson::object();
    size_t tokens = 0;
    std::set<std::string> counted;
    for (const auto& key : request.keys) {
        auto e = request.errors.find(key);
        if (e != request.errors.end()) {
            answers[key] = e->second;
            continue;
        }
        const auto rows = request.rows_of(key);
        if (rows.empty()) throw std::runtime_error("no row for question " + key);
        const ComposedRow& row = request.rows[rows[0]];
        try {
            answers[key] = row_answer(row, values.at(rows[0]), config);
            // upstream counts a question's tokens once it has answered: each distinct state once, every block
            if (counted.insert(row.state_key).second) tokens += static_cast<size_t>(row.state_len);
            tokens += row.ids.size() - static_cast<size_t>(row.state_len);
        } catch (const ContractError& ex) {
            answers[key] = caught(row.type, "invalid_question", ex.what());
        }
    }
    add_reasons(answers);
    ojson out = ojson::object();
    out["model"] = config.model_name;
    out["answers"] = answers;
    out["usage"] = {{"input_tokens", tokens}, {"output_tokens", 0}};
    return out;
}

ojson run_composed(const ComposedRequest& request, ReadoutEngine& engine, const float* readout, const ComposedConfig& config,
                   const std::function<EngineImage(size_t)>& image) {
    std::vector<std::vector<std::vector<double>>> values(request.rows.size());
    for (const auto& group : state_groups(request)) {
        const EngineStateRequest call = build_engine_request(request, group, image);
        const auto hidden = engine.encode_state_once(call);
        if (hidden.size() != group.size()) throw std::runtime_error("the engine returned a different number of blocks than asked for");
        for (size_t i = 0; i < group.size(); i++) {
            values[group[i]] = row_probabilities(request.rows[group[i]], hidden[i], readout, engine.hidden_size(), config.temperature);
        }
    }
    return respond_composed(request, values, config);
}

}  // namespace flm::d3
