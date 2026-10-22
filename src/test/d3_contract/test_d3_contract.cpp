/// \file test_d3_contract.cpp
/// \brief Contract test for the d3 decision model (no NPU): replays the fixtures that gen_fixtures.py
///        writes with the upstream Python package through src/common/decision and requires
///        byte-identical prompts, identical request/question errors and identical answer objects.
///
///   test_d3_contract <fixtures dir> [<D3-Lite-NPU2 package dir>]
///
/// With a package dir (and FLM_D3_TEST_TOKENIZER), token ids and the 255 answer codes are checked
/// against tokenizer.json too.
#include <cmath>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

#include "decision/d3_contract.hpp"
#include "decision/py_json.hpp"

#ifdef FLM_D3_TEST_TOKENIZER
#include "tokenizers_cpp.h"
#endif

using flm::d3::ojson;
using flm::d3::PyValue;

namespace {

int failures = 0;
int checks = 0;

void fail(const std::string& id, const std::string& what) {
    if (failures < 40) std::cerr << "FAIL " << id << ": " << what << "\n";
    failures++;
}

/// Structural equality; numbers must match to `tol` relative (0 = exactly).
bool same(const ojson& a, const ojson& b, double tol, std::string& where) {
    if (a.is_number() && b.is_number()) {
        double x = a.get<double>(), y = b.get<double>();
        if (x == y) return true;
        if (tol > 0 && std::fabs(x - y) <= tol * std::max(std::fabs(x), std::fabs(y))) return true;
        char buf[128];
        std::snprintf(buf, sizeof(buf), "%.17g != %.17g", x, y);
        where = buf;
        return false;
    }
    if (a.type() != b.type()) {
        where = "type " + a.dump() + " vs " + b.dump();
        return false;
    }
    if (a.is_object()) {
        if (a.size() != b.size()) { where = "size " + a.dump() + " vs " + b.dump(); return false; }
        auto ia = a.begin();
        auto ib = b.begin();
        for (; ia != a.end(); ++ia, ++ib) {
            if (ia.key() != ib.key()) { where = "key order " + ia.key() + " vs " + ib.key(); return false; }
            if (!same(ia.value(), ib.value(), tol, where)) { where = ia.key() + "." + where; return false; }
        }
        return true;
    }
    if (a.is_array()) {
        if (a.size() != b.size()) { where = "length"; return false; }
        for (size_t i = 0; i < a.size(); i++)
            if (!same(a[i], b[i], tol, where)) return false;
        return true;
    }
    if (a != b) { where = a.dump() + " vs " + b.dump(); return false; }
    return true;
}

std::string read_file(const std::string& p) {
    std::ifstream f(p, std::ios::binary);
    std::stringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "usage: test_d3_contract <fixtures dir> [<package dir>]\n";
        return 2;
    }
    const std::string dir = argv[1];
    const ojson config = ojson::parse(read_file(dir + "/decision_config.json"));
    const std::vector<std::string> codes = config["codes"].get<std::vector<std::string>>();
    const std::vector<int> token_ids = config["token_ids"].get<std::vector<int>>();

    std::function<std::vector<int>(const std::string&)> encode;
#ifdef FLM_D3_TEST_TOKENIZER
    std::unique_ptr<tokenizers::Tokenizer> tok;
    if (argc > 2) {
        tok = tokenizers::Tokenizer::FromBlobJSON(read_file(std::string(argv[2]) + "/tokenizer.json"));
        encode = [&](const std::string& s) { return tok->Encode(s); };
        std::vector<std::string> c2;
        std::vector<int> t2;
        flm::d3::answer_codes(encode, c2, t2);
        checks++;
        if (c2 != codes || t2 != token_ids) fail("answer_codes", "codes/token ids differ from decision_config.json");
    }
#endif
    // the candidate codes are a superset in preference order
    {
        auto cand = flm::d3::candidate_codes();
        checks++;
        if (cand.size() != 26 + 26 * 26 || cand[26] != "AA") fail("candidate_codes", "unexpected list");
    }

    int records = 0, prompts = 0, token_checks = 0, answers = 0, request_errors = 0, parse_errors = 0;
    int parse_message_diffs = 0;
    for (const char* file : {"goldens.jsonl", "fuzz.jsonl"}) {
        std::ifstream f(dir + "/" + file);
        if (!f) { fail(file, "missing"); continue; }
        std::string line;
        while (std::getline(f, line)) {
            if (line.empty()) continue;
            const ojson rec = ojson::parse(line);
            const std::string id = rec["id"];
            const std::string body_text = rec["body"];
            records++;

            PyValue body;
            try {
                body = flm::d3::parse_json(body_text);
            } catch (const flm::d3::JsonError& e) {
                checks++;
                parse_errors++;
                if (!rec.value("parse_error", false)) { fail(id, std::string("unexpected parse error: ") + e.what()); continue; }
                const std::string ours = std::string("could not read the body as JSON (") + e.what() + ")";
                if (ours != rec["detail"].get<std::string>()) parse_message_diffs++;
                continue;
            }
            if (rec.value("parse_error", false)) { checks++; fail(id, "body parsed but upstream could not"); continue; }

            flm::d3::Prepared p;
            try {
                p = flm::d3::prepare(body, codes);
            } catch (const flm::d3::RequestError& e) {
                checks++;
                request_errors++;
                std::string where;
                if (!rec.contains("detail") || !rec["detail"].is_string()) fail(id, std::string("unexpected request error: ") + e.what());
                else if (!same(e.detail, rec["detail"], 0, where)) fail(id, "request error " + e.detail.dump() + " vs " + rec["detail"].dump());
                continue;
            }
            if (rec.contains("detail") && rec["detail"].is_string()) {
                checks++;
                fail(id, "upstream refused the request: " + rec["detail"].dump());
                continue;
            }

            // prompts: byte-identical
            for (auto it = rec["texts"].begin(); it != rec["texts"].end(); ++it) {
                checks++;
                prompts++;
                auto t = p.texts.find(it.key());
                if (t == p.texts.end()) fail(id, "no prompt for " + it.key());
                else if (t->second != it.value().get<std::string>()) fail(id, "prompt bytes differ for " + it.key());
            }
            // token ids: from the tokenizer when available, else taken from the fixture
            if (encode) {
                flm::d3::tokenize(p, encode, 0);
                for (auto it = rec["ids"].begin(); it != rec["ids"].end(); ++it) {
                    checks++;
                    token_checks++;
                    if (p.sequences[it.key()] != it.value().get<std::vector<int>>()) fail(id, "token ids differ for " + it.key());
                }
            } else {
                for (auto it = rec["ids"].begin(); it != rec["ids"].end(); ++it)
                    p.sequences[it.key()] = it.value().get<std::vector<int>>();
            }
            // invalid_questions (the 422 the server sends) or none; goldens are in-process replies
            if (std::string(file) == "fuzz.jsonl") {
                checks++;
                ojson inv = flm::d3::invalid_questions(p);
                ojson want = rec.contains("detail") ? rec["detail"]["invalid_questions"] : ojson::object();
                std::string where;
                if (!same(inv, want, 0, where)) fail(id, "invalid_questions: " + where);
            }
            // the in-process response from the same fp32 probabilities
            std::map<std::string, std::vector<double>> probs;
            for (auto it = rec["probs"].begin(); it != rec["probs"].end(); ++it)
                probs[it.key()] = it.value().get<std::vector<double>>();
            ojson resp = flm::d3::respond(p, probs, rec["response"]["model"].get<std::string>());
            checks++;
            answers += static_cast<int>(rec["response"]["answers"].size());
            std::string where;
            if (!same(resp, rec["response"], 0, where)) {
                std::string w2;
                if (same(resp, rec["response"], 1e-15, w2)) fail(id, "response within 1e-15 but not exact: " + where);
                else fail(id, "response: " + where);
            }
        }
    }

    // images.jsonl: image requests (strict data URLs) -> 422 texts, prompts, grids, expanded token ids
    int image_records = 0;
    {
        // PIL's unidentified-image text carries an object address ("<_io.BytesIO object at 0x...>")
        const auto unaddressed = [](std::string s) {
            const size_t at = s.find(" <_io.BytesIO object at 0x");
            if (at != std::string::npos) s.erase(at, s.find('>', at) + 1 - at);
            return s;
        };
        std::ifstream f(dir + "/images.jsonl");
        if (!f) fail("images.jsonl", "missing");
        std::string line;
        while (std::getline(f, line)) {
            if (line.empty()) continue;
            const ojson rec = ojson::parse(line);
            const std::string id = rec["id"];
            records++;
            image_records++;
            const PyValue body = flm::d3::parse_json(rec["body"].get<std::string>());
            const bool refused = rec.contains("detail") && rec["detail"].is_string();
            const bool decode_stage = rec.value("stage", std::string()) == "decode";
            if (decode_stage) {
                // the server's decoder refuses image i: stand in for it, and the refusal must name image i
                // (images after it are not looked at, as upstream loads them in order)
                const std::string want = rec["detail"].get<std::string>();
                const size_t bad = std::stoul(want.substr(want.find('[') + 1));
                std::string got;
                try {
                    flm::d3::prepare(body, codes, [bad](size_t i, const flm::d3::ImageFile&) {
                        if (i == bad) throw flm::d3::ContractError("the image could not be decoded (test)");
                    });
                } catch (const flm::d3::RequestError& e) {
                    got = e.detail.get<std::string>();
                }
                checks++;
                request_errors++;
                if (got != "images[" + std::to_string(bad) + "]: the image could not be decoded (test)")
                    fail(id, "decode refusal '" + got + "' vs upstream '" + want + "'");
                continue;
            }
            flm::d3::Prepared p;
            try {
                p = flm::d3::prepare(body, codes);
            } catch (const flm::d3::RequestError& e) {
                checks++;
                request_errors++;
                if (!refused) fail(id, std::string("unexpected request error: ") + e.what());
                else if (e.detail.get<std::string>() != unaddressed(rec["detail"].get<std::string>()))
                    fail(id, "request error '" + e.detail.get<std::string>() + "' vs '" + rec["detail"].get<std::string>() + "'");
                continue;
            }
            if (refused) {
                checks++;
                fail(id, "upstream refused the request: " + rec["detail"].dump());
                continue;
            }
            {
                checks++;
                ojson inv = flm::d3::invalid_questions(p);
                ojson want = rec.contains("detail") ? rec["detail"]["invalid_questions"] : ojson::object();
                std::string where;
                if (!same(inv, want, 0, where)) fail(id, "invalid_questions: " + where);
            }
            for (auto it = rec["texts"].begin(); it != rec["texts"].end(); ++it) {
                checks++;
                prompts++;
                auto t = p.texts.find(it.key());
                if (t == p.texts.end()) fail(id, "no prompt for " + it.key());
                else if (t->second != it.value().get<std::string>()) fail(id, "prompt bytes differ for " + it.key());
            }
            if (rec.contains("grids")) {
                checks++;
                ojson grids = ojson::array();
                for (const auto& g : p.image_grids) grids.push_back({1, g.first, g.second});
                std::string where;
                if (!same(grids, rec["grids"], 0, where)) fail(id, "grids " + grids.dump() + " vs " + rec["grids"].dump());
            }
            if (encode) {
                flm::d3::tokenize(p, encode, 0);
                for (auto it = rec["ids"].begin(); it != rec["ids"].end(); ++it) {
                    checks++;
                    token_checks++;
                    if (p.sequences[it.key()] != it.value().get<std::vector<int>>()) fail(id, "token ids differ for " + it.key());
                }
                if (rec.contains("max_length_2000")) {
                    flm::d3::Prepared q = flm::d3::prepare(body, codes);
                    flm::d3::tokenize(q, encode, 2000);
                    ojson got = ojson::object();
                    for (const auto& k : q.keys) {
                        auto e = q.errors.find(k);
                        if (e != q.errors.end() && e->second["error"] == "max_length_exceeded") got[k] = e->second["message"];
                    }
                    checks++;
                    std::string where;
                    if (!same(got, rec["max_length_2000"], 0, where)) fail(id, "max_length_exceeded: " + where);
                }
            }
        }
    }

    // image checks the fixtures cannot carry (sizes) or that are FastFlowLM's own
    {
        const auto detail = [&](const std::string& body_json) -> std::string {
            try {
                flm::d3::prepare(flm::d3::parse_json(body_json), codes);
            } catch (const flm::d3::RequestError& e) {
                return e.detail.get<std::string>();
            }
            return "";
        };
        const std::string q = "\"questions\": {\"n\": {\"type\": \"noul\"}}";
        struct Case { std::string body; std::string want; };
        const std::vector<Case> cases = {
            // _decode_base64: the encoded text is refused unread past 4 * ceil(limit / 3) characters
            {"{" + q + ", \"images\": [\"data:image/png;base64," + std::string(10666669, 'A') + "\"]}",
             "images[0]: the encoded data exceeds the 8,000,000 byte limit"},
            {"{" + q + ", \"images\": [\"data:image/png;base64," + std::string(10666668, 'A') + "\"]}",
             "images[0]: the decoded data has 8,000,001 bytes, over the 8,000,000 byte limit"},
            // videos stay unavailable (images first, as MEDIA_FIELDS orders them)
            {"{" + q + ", \"videos\": [\"data:video/mp4;base64,AAAA\"]}",
             "video inputs are not available: this FastFlowLM build runs d3 on text and images"},
            {"{" + q + ", \"images\": [7], \"videos\": [\"x\"]}",
             "images[0]: strict mode: images must be given as a data: URL string, not a file path, web address or PIL image"},
        };
        for (const auto& c : cases) {
            checks++;
            const std::string got = detail(c.body);
            if (got != c.want) fail("image-limits", "'" + got.substr(0, 120) + "' vs '" + c.want + "'");
        }
        // smart_resize rounds half to even, as Python's round() (80 / 32 = 2.5 -> 2)
        checks++;
        if (flm::d3::smart_resize(80, 112, 32, 0, 1 << 30) != std::make_pair(64, 128)) fail("smart_resize", "half-even");
        // prepare()'s decode callback runs per image right after its inspection (upstream's order)
        {
            const std::string png = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAIAAACQd1PeAAAADElEQVR4nGNgYGAAAAAEAAH2FzhVAAAAAElFTkSuQmCC";
            const std::string body = "{" + q + ", \"images\": [\"" + png + "\", \"data:image/gif;base64,AAAA\"]}";
            std::string got;
            try {
                flm::d3::prepare(flm::d3::parse_json(body), codes, [](size_t i, const flm::d3::ImageFile&) {
                    if (i == 0) throw flm::d3::ContractError("the image could not be decoded (test)");
                });
            } catch (const flm::d3::RequestError& e) {
                got = e.detail.get<std::string>();
            }
            checks++;
            if (got != "images[0]: the image could not be decoded (test)") fail("decode-order", got);
        }
    }

    // json.loads(bytes): encoding detection and UnicodeDecodeError texts (from CPython 3.12)
    {
        struct Case { std::string body; std::string want; };   // want: "" = parses
        const std::vector<Case> cases = {
            {"\xef\xbb\xbf{\"a\": 1}", ""},
            {std::string("{\"a\": 1}", 8), ""},
            {"{\"a\": \"x\xffy\"}", "'utf-8' codec can't decode byte 0xff in position 8: invalid start byte"},
            {"{\"a\": \"\xe2\x28\xa1\"}", "'utf-8' codec can't decode byte 0xe2 in position 7: invalid continuation byte"},
            {"{\"a\": \"x\"}\xe2\x82", "'utf-8' codec can't decode bytes in position 10-11: unexpected end of data"},
            {"{\"a\": 1}\xc3", "'utf-8' codec can't decode byte 0xc3 in position 8: unexpected end of data"},
            {"{\"a\": \"\xc0\xaf\"}", "'utf-8' codec can't decode byte 0xc0 in position 7: invalid start byte"},
            {"{\"a\": \"\xe0\x80\x80\"}", "'utf-8' codec can't decode byte 0xe0 in position 7: invalid continuation byte"},
            {"\xe2\x82\x28", "'utf-8' codec can't decode bytes in position 0-1: invalid continuation byte"},
            {"\xf0\x9f\x98\x28", "'utf-8' codec can't decode bytes in position 0-2: invalid continuation byte"},
            {"\xf4\x90\x80\x80", "'utf-8' codec can't decode byte 0xf4 in position 0: invalid continuation byte"},
            {std::string("\xff\xfe{\0\"\0a\0\"\0:\0 \0""1\0}\0", 18), ""},   // UTF-16-LE with BOM
        };
        for (const auto& c : cases) {
            checks++;
            std::string got;
            try {
                flm::d3::parse_json(c.body);
            } catch (const flm::d3::JsonError& e) {
                got = e.what();
            }
            if (got != c.want) fail("bytes", "'" + got + "' vs '" + c.want + "'");
        }
    }

    // the readout against a hand-computed case
    {
        float hidden[4] = {1.0f, -2.0f, 0.5f, 0.0f};
        float w[3 * 4] = {1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0};
        auto pr = flm::d3::readout_probabilities(hidden, w, 4, 2, 1.0f);
        checks++;
        double e0 = std::exp(1.0), e1 = std::exp(-2.0);
        if (pr.size() != 2 || std::fabs(pr[0] - e0 / (e0 + e1)) > 1e-6) fail("readout", "softmax");
    }

    std::cout << "d3 contract: " << records << " requests (" << image_records << " with images), " << prompts << " prompts, " << token_checks
              << " token-id checks, " << answers << " answers, " << request_errors << " request errors, "
              << parse_errors << " parse errors (" << parse_message_diffs << " with a different message); "
              << checks << " checks, " << failures << " failures\n";
    return failures == 0 ? 0 : 1;
}
