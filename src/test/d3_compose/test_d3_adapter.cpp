/// \file test_d3_adapter.cpp
/// \brief The engine adapter's per-row (M1) path against a mock `decision_model`: which prompts and payloads reach
///        `encode_batch` for blocks with and without own images, and what it refuses. CPU only. The state-once path
///        (`d3_npu::encode_state_once`) needs the real library and is only compile-checked here.
#include <iostream>

#include "AutoDecisionModel/d3_engine_adapter.hpp"
#include "decision/d3_engine.hpp"
#include "models/d3/flm/aie2p/d3_npu.hpp"

// The device runtime is not linked: buffers here live on the host, so the out-of-line destructor of a device
// buffer object (never created) only has to exist.
#if __has_include("xrt/xrt_bo.h")
namespace xrt { bo::~bo() = default; }
#endif

namespace {
int failures = 0, checks = 0;
void expect(bool ok, const char* what) {
    checks++;
    if (!ok) { failures++; std::cerr << "FAIL " << what << "\n"; }
}

struct Call {
    std::vector<std::vector<int>> ids;
    std::vector<int> state_len;
    std::vector<std::pair<int, int>> grids;
    size_t pixels = 0;
};

class MockEngine : public decision_model {
public:
    std::vector<Call> calls;
    void load_weights(Q4NX&) override {}
    std::vector<buffer<bf16>> encode_batch(const std::vector<std::vector<int>>& ids, void* payload) override {
        Call c;
        c.ids = ids;
        auto* p = static_cast<d3_image_payload*>(payload);
        c.state_len = p->state_len;
        for (const auto& im : p->images) c.grids.emplace_back(im.grid_h, im.grid_w);
        c.pixels = p->pixels.size();
        calls.push_back(c);
        std::vector<buffer<bf16>> out;
        for (const auto& row : ids) {
            buffer<bf16> h(4);
            for (int d = 0; d < 4; d++) h.data()[d] = bf16(static_cast<float>(row.back() + d));   // last token id + d
            out.push_back(std::move(h));
        }
        return out;
    }
    buffer<bf16> encode(const std::vector<int>&, void*) override { return buffer<bf16>(4); }
    int max_length() override { return 32768; }
    int hidden_size() override { return 4; }
};
}  // namespace

int main() {
    MockEngine mock;
    auto adapter = make_d3_engine_adapter(&mock, nullptr, 0, true);
    flm::d3::EngineStateRequest req;
    req.image_token_id = 99;
    req.state_ids = {1, 2, 3};
    req.state_image_ids = {0};
    flm::d3::EngineImage a, b;
    a.grid_h = a.grid_w = 2; a.pixels.assign(4 * 1536, 1);
    b.grid_h = 2; b.grid_w = 4; b.pixels.assign(8 * 1536, 2);
    req.images = {a, b};
    flm::d3::EngineBlock q0{"q0", {10, 11}, {0}, {}};
    flm::d3::EngineBlock q1{"q1", {20, 99}, {0}, {1}};
    flm::d3::EngineBlock q2{"q2", {30, 31}, {0}, {}};
    req.blocks = {q0, q1, q2};
    const auto rows = adapter->encode_state_once(req);
    expect(rows.size() == 3 && rows[0].size() == 4 && rows[0][0] == 11 && rows[1][0] == 99 && rows[2][0] == 31, "rows in block order");
    // two batches: blocks without own images share the state image only; the block with an image has its own payload
    expect(mock.calls.size() == 2, "one encode_batch per distinct image set");
    bool plain = false, with_image = false;
    for (const auto& c : mock.calls) {
        if (c.ids.size() == 2 && c.grids.size() == 1 && c.grids[0] == std::make_pair(2, 2) && c.state_len == std::vector<int>{3, 3} &&
            c.ids[0] == std::vector<int>{1, 2, 3, 10, 11} && c.ids[1] == std::vector<int>{1, 2, 3, 30, 31} && c.pixels == 4 * 1536)
            plain = true;
        if (c.ids.size() == 1 && c.grids.size() == 2 && c.grids[1] == std::make_pair(2, 4) && c.state_len == std::vector<int>{3} && c.pixels == 12 * 1536)
            with_image = true;
    }
    expect(plain && with_image, "payloads: state image first, then the block's images; state_len = |state|");
    // a Box block (four readout rows) is refused by the per-row engine, with the reason in the message
    req.blocks = {flm::d3::EngineBlock{"box", {40, 41, 42, 43, 44, 45, 46, 47, 48, 49}, {9, 6, 3, 0}, {}}};
    std::string what;
    try { adapter->encode_state_once(req); } catch (const std::runtime_error& e) { what = e.what(); }
    expect(what.find("Box question reads four") != std::string::npos, "box refused without state-once");
    std::cout << "d3 adapter: " << checks << " checks, " << failures << " failures\n";
    return failures == 0 ? 0 : 1;
}
