/// \file test_d3_golden_rows.cpp
/// \brief The composed requests of the frozen M3/M4 goldens (box_image, assets, video, box_video, track) through
///        prepare_composed with FastFlowLM's own tokenizer and the FFmpeg decoder, against the golden rows: the rendered
///        prompt text, the token count n, the state block length S, the readout offsets, per-question inline errors
///        and `usage.input_tokens` (from uniform probabilities). CPU only; optional (needs the goldens, the d3rig request
///        files, the package's tokenizer and the FFmpeg libraries; build with /usr/bin/g++, see CMakeLists.txt).
///
///   test_d3_golden_rows <package dir> <goldens dir> <requests dir> <set> [id-substring]
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>

#include "decision/d3_compose.hpp"
#include "decision/d3_composed.hpp"
#include "tokenizer/tokenizer.hpp"
#include "video/video_input.hpp"

using flm::d3::ojson;
using flm::d3::PyValue;
namespace v = flm::d3::video;
namespace fs = std::filesystem;

namespace {

std::string read_file(const fs::path& p) {
    std::ifstream f(p, std::ios::binary);
    std::stringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

std::string base64(const std::string& in) {
    static const char* tbl = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    std::string out;
    for (size_t i = 0; i < in.size(); i += 3) {
        const uint32_t n = (static_cast<uint8_t>(in[i]) << 16) | (i + 1 < in.size() ? static_cast<uint8_t>(in[i + 1]) << 8 : 0) |
                           (i + 2 < in.size() ? static_cast<uint8_t>(in[i + 2]) : 0);
        out += tbl[n >> 18];
        out += tbl[(n >> 12) & 63];
        out += i + 1 < in.size() ? tbl[(n >> 6) & 63] : '=';
        out += i + 2 < in.size() ? tbl[n & 63] : '=';
    }
    return out;
}

/// A request path (under the requests dir) -> a data URL of its bytes; anything else unchanged.
ojson media(const ojson& item, const fs::path& dir) {
    if (!item.is_string() || item.get<std::string>().rfind("data:", 0) == 0) return item;
    const fs::path p = dir / item.get<std::string>();
    if (!fs::is_regular_file(p)) return item;
    const std::string ext = p.extension().string();
    const std::string mime = ext == ".png" ? "image/png" : ext == ".jpg" || ext == ".jpeg" ? "image/jpeg" : ext == ".webp" ? "image/webp"
                             : ext == ".mp4" ? "video/mp4" : ext == ".webm" ? "video/webm" : ext == ".mov" ? "video/quicktime" : "video/x-matroska";
    return "data:" + mime + ";base64," + base64(read_file(p));
}

std::vector<ojson> read_jsonl(const fs::path& p) {
    std::vector<ojson> out;
    std::ifstream f(p);
    std::string line;
    while (std::getline(f, line)) {
        if (!line.empty()) out.push_back(ojson::parse(line));
    }
    return out;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 5) {
        std::cerr << "usage: test_d3_golden_rows <package dir> <goldens dir> <requests dir> <set> [id substring]\n";
        return 2;
    }
    const fs::path package = argv[1], goldens = fs::path(argv[2]) / argv[4], requests = argv[3];
    const std::string only = argc > 5 ? argv[5] : "";
    Tokenizer tokenizer(package.string());
    const ojson config = ojson::parse(read_file(package / "decision_config.json"));
    const std::vector<std::string> codes = config["codes"].get<std::vector<std::string>>();
    flm::d3::ComposedConfig cfg;
    for (const auto& k : config["systemone"]["box_calibration"]) cfg.box_calibration.emplace_back(k[0].get<double>(), k[1].get<double>());
    cfg.video = true;
    cfg.refuse_video_questions = false;
    cfg.moment_row_limit = 100000;
    flm::d3::ComposedHooks hooks;
    hooks.encode = [&](const std::string& s) { return tokenizer.encode(s); };
    hooks.token_bytes = [&](int id) { return tokenizer.token_byte_length(id); };
    v::VideoLoadOptions load;
    load.backend = v::make_ffmpeg_backend();
    std::string why;
    if (!load.backend->available(&why)) {
        std::cerr << "FFmpeg is not available: " << why << "\n";
        return 2;
    }
    hooks.load_video = v::make_video_loader(load);
    hooks.load_windows = [&](const PyValue& value, int rate) { return v::load_video_windows(value, rate, load); };

    std::map<std::string, std::vector<ojson>> golden_rows;
    for (const auto& r : read_jsonl(goldens / "rows.jsonl")) golden_rows[r["id"].get<std::string>()].push_back(r);
    int failures = 0, checks = 0, requests_run = 0, rows_checked = 0;
    const auto fail = [&](const std::string& id, const std::string& what) {
        failures++;
        std::cerr << "FAIL " << id << ": " << what << "\n";
    };
    for (const auto& answer : read_jsonl(goldens / "answers.jsonl")) {
        const std::string id = answer["id"].get<std::string>();
        if (!only.empty() && id.find(only) == std::string::npos) continue;
        ojson row;
        for (const auto& r : read_jsonl(requests / (std::string(argv[4]) + ".jsonl"))) {
            if (r["id"] == id) { row = r; break; }
        }
        if (row.is_null()) { fail(id, "no request"); continue; }
        ojson body = row["request"];
        for (const char* field : {"images", "videos"}) {
            if (body.contains(field) && body[field].is_array()) {
                for (auto& item : body[field]) item = media(item, requests);
            }
        }
        if (body.contains("assets") && body["assets"].is_object()) {
            for (auto& item : body["assets"]) item = media(item, requests);
        }
        requests_run++;
        flm::d3::ComposedRequest req;
        const auto began = std::chrono::steady_clock::now();
        try {
            req = flm::d3::prepare_composed(flm::d3::parse_json(body.dump()), codes, cfg, hooks);
        } catch (const flm::d3::RequestError& e) {
            checks++;
            if (answer.contains("response")) fail(id, std::string("refused with 422 ") + e.what() + ", the golden answered");
            continue;
        }
        const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - began).count();
        checks++;
        if (!answer.contains("response")) { fail(id, "answered, the golden refused the request"); continue; }
        // rows: in request order per question; golden rows are in the same order (question, moment)
        const auto& want = golden_rows[id];
        if (want.size() != req.rows.size()) {
            fail(id, "rows " + std::to_string(req.rows.size()) + " vs golden " + std::to_string(want.size()));
            continue;
        }
        int bad = 0;
        for (size_t i = 0; i < want.size(); i++) {
            const auto& g = want[i];
            const auto& r = req.rows[i];
            rows_checked++;
            checks++;
            const bool same_key = g["key"] == r.key;
            const bool same = same_key && g["state"].get<int>() == r.state_len && g["n"].get<int>() == static_cast<int>(r.ids.size()) &&
                              g["offsets"].get<std::vector<int>>() == r.readout.offsets && g["counts"].get<std::vector<int>>() == r.readout.counts;
            if (!same) {
                bad++;
                if (bad <= 3) {
                    fail(id, "row " + std::to_string(i) + " key " + r.key + ": S " + std::to_string(r.state_len) + "/" + g["state"].dump() + ", n " +
                                 std::to_string(r.ids.size()) + "/" + g["n"].dump() + ", offsets " + ojson(r.readout.offsets).dump() + "/" + g["offsets"].dump());
                }
                continue;
            }
            checks++;
            // the rendered prompt: the golden stores it with the placeholders the processor expands
            if (g.contains("text") && g["text"].is_string() && g["text"].get<std::string>() != r.text) {
                bad++;
                if (bad <= 3) fail(id, "row " + std::to_string(i) + " text differs");
            }
        }
        // inline errors and usage (the engine's values do not matter for them: uniform probabilities)
        std::vector<std::vector<std::vector<double>>> values(req.rows.size());
        for (size_t i = 0; i < req.rows.size(); i++) {
            for (size_t p = 0; p < req.rows[i].readout.counts.size(); p++)
                values[i].push_back(std::vector<double>(req.rows[i].readout.counts[p], 1.0 / req.rows[i].readout.counts[p]));
        }
        const ojson ours = flm::d3::respond_composed(req, values, cfg);
        const ojson& golden_response = answer["response"];
        checks++;
        if (ours["usage"] != golden_response["usage"]) {
            bad++;
            fail(id, "usage " + ours["usage"].dump() + " vs " + golden_response["usage"].dump());
        }
        for (auto it = golden_response["answers"].begin(); it != golden_response["answers"].end(); ++it) {
            if (!it.value().contains("error")) continue;
            checks++;
            if (!ours["answers"].contains(it.key()) || ours["answers"][it.key()] != it.value()) {
                bad++;
                fail(id, "error answer " + it.key() + ": " + (ours["answers"].contains(it.key()) ? ours["answers"][it.key()].dump() : "none") + " vs " + it.value().dump());
            }
        }
        std::cout << (bad ? "BAD  " : "ok   ") << id << " rows " << want.size() << " (" << seconds << " s)\n" << std::flush;
    }
    std::cout << "golden rows (" << argv[4] << "): " << requests_run << " requests, " << rows_checked << " rows, " << checks << " checks, " << failures << " failures\n";
    return failures == 0 ? 0 : 1;
}
