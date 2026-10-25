/// \file test_prefix_slot_cache.cpp
/// \brief PrefixSlotCache against a fake engine: which slot a prompt resumes from, what is evicted, and that
///        engine slot 0 (the model's single checkpoint()/restore()) is never handed out. No NPU needed.
#include "prefix_slot_cache.hpp"

#include <cstdio>
#include <map>
#include <numeric>
#include <string>
#include <vector>

namespace {
int failures = 0;
#define CHECK(cond)                                                                 \
    do {                                                                            \
        if (!(cond)) {                                                              \
            std::printf("  FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond);           \
            failures++;                                                             \
        }                                                                           \
    } while (0)

/// \brief records what the cache asks of the engine; `length` plays the engine's context length
struct fake_engine {
    struct slot { int length = 0; int kv_from = 0; };
    std::map<int, slot> slots;
    std::vector<std::string> calls;
    int length = 0;
    int max_slots = 33;
    size_t row_bytes = 1024;            // 1 KiB per row
    size_t state_bytes = 1u << 20;      // 1 MiB per snapshot
    bool fail_restore = false;
};

class fake_ops final : public prefix_engine_ops {
public:
    explicit fake_ops(fake_engine* e) : e_(e) {}
    int checkpoint_slot(int slot, int kv_from) override {
        if (slot < 0 || slot >= e_->max_slots) return -1;
        e_->slots[slot] = {e_->length, kv_from};
        e_->calls.push_back("save " + std::to_string(slot));
        return e_->length;
    }
    int restore_slot(int slot) override {
        e_->calls.push_back("restore " + std::to_string(slot));
        auto it = e_->slots.find(slot);
        if (e_->fail_restore || it == e_->slots.end()) return -1;
        e_->length = it->second.length;
        return e_->length;
    }
    int restore_slot_kv(int slot) override {
        e_->calls.push_back("restore_kv " + std::to_string(slot));
        auto it = e_->slots.find(slot);
        return it == e_->slots.end() ? -1 : it->second.length;
    }
    void drop_checkpoint_slot(int slot) override {
        e_->calls.push_back("drop " + std::to_string(slot));
        e_->slots.erase(slot);
    }
    size_t row_bytes() override { return e_->row_bytes; }
    size_t state_bytes() override { return e_->state_bytes; }
    int max_slots() override { return e_->max_slots; }

private:
    fake_engine* e_;
};

std::vector<int> seq(int from, int count) {
    std::vector<int> v(count);
    std::iota(v.begin(), v.end(), from);
    return v;
}
std::vector<int> cat(std::vector<int> a, const std::vector<int>& b) {
    a.insert(a.end(), b.begin(), b.end());
    return a;
}

/// \brief what AutoModel::_prefix_insert does after a prefill: boundary snapshot at `head`, end snapshot at the end
void prefill(PrefixSlotCache& cache, fake_engine& e, const std::vector<int>& prompt, size_t head) {
    if (head > 0) {
        std::vector<int> h(prompt.begin(), prompt.begin() + head);
        e.length = static_cast<int>(head);
        cache.set_live(h);
        cache.save(h, {}, true, 0);
    }
    e.length = static_cast<int>(prompt.size());
    cache.set_live(prompt);
    cache.save(prompt, {}, false, cache.end_kv_from(prompt, head));
}

size_t resume(PrefixSlotCache& cache, const std::vector<int>& prompt, const std::vector<uint64_t>& images = {}) {
    std::vector<int> matched;
    const auto r = cache.resume(prompt, images, matched);
    return r.resumed ? r.match_len : 0;
}

PrefixSlotCache make(fake_engine& e, int mb = -1, int slots = -1) {
    PrefixSlotCache cache;
    cache.configure(mb, slots);
    cache.attach(std::make_unique<fake_ops>(&e));
    return cache;
}

void test_slot_zero_reserved() {
    std::printf("slot 0 is never handed out; at most 32 slots\n");
    fake_engine e;
    PrefixSlotCache cache = make(e, 100000, 40);
    for (int i = 0; i < 40; i++) {
        prefill(cache, e, seq(1000 * (i + 1), 10), 0);
    }
    CHECK(e.slots.count(0) == 0);
    CHECK(cache.slot_count() == 32);
    // an engine built with only 32 slots (older lib) leaves 31
    fake_engine old;
    old.max_slots = 32;
    PrefixSlotCache cache_old = make(old, 100000, 32);
    for (int i = 0; i < 40; i++) {
        prefill(cache_old, old, seq(1000 * (i + 1), 10), 0);
    }
    CHECK(old.slots.count(0) == 0);
    CHECK(cache_old.slot_count() == 31);
}

void test_longest_strict_prefix() {
    std::printf("resumes from the longest saved prefix that is strictly shorter than the prompt\n");
    fake_engine e;
    PrefixSlotCache cache = make(e);
    const auto sys = seq(1, 600);
    const auto t1 = cat(sys, seq(5000, 100));
    prefill(cache, e, t1, 600);
    const auto t2 = cat(t1, seq(6000, 30));
    CHECK(resume(cache, t2) == t1.size());
    // the same prompt again: the end snapshot is not strictly shorter, the boundary one is
    CHECK(resume(cache, t1) == sys.size());
    // a different conversation with the same system prompt resumes at the boundary
    CHECK(resume(cache, cat(sys, seq(7000, 40))) == sys.size());
    // a different system prompt matches nothing
    CHECK(resume(cache, seq(9000, 700)) == 0);
}

void test_head_restored_after_cold_request() {
    std::printf("an end snapshot gets its head rows from the boundary snapshot once the live cache lost them\n");
    fake_engine e;
    PrefixSlotCache cache = make(e);
    const auto sys = seq(1, 600);
    const auto t1 = cat(sys, seq(5000, 100));
    prefill(cache, e, t1, 600);
    // a side request with another system prompt (a title call) empties the live cache
    CHECK(resume(cache, seq(9000, 100)) == 0);
    cache.clear_live();
    e.calls.clear();
    CHECK(resume(cache, cat(t1, seq(6000, 30))) == t1.size());
    CHECK(e.calls.size() == 2 && e.calls[0].rfind("restore_kv ", 0) == 0 && e.calls[1].rfind("restore ", 0) == 0);
    // with the live cache intact, no head restore is needed
    e.calls.clear();
    CHECK(resume(cache, cat(t1, seq(6100, 30))) == t1.size());
    CHECK(e.calls.size() == 1);
}

void test_eviction_order() {
    std::printf("eviction: superseded end snapshots, then LRU end snapshots, then boundary snapshots without dependents\n");
    fake_engine e;
    PrefixSlotCache cache = make(e, 100000, 4);
    const auto sys_a = seq(1, 600);
    const auto a1 = cat(sys_a, seq(5000, 100));
    prefill(cache, e, a1, 600);                       // B_a, E_a1
    const auto a2 = cat(a1, seq(6000, 100));
    CHECK(resume(cache, a2) == a1.size());
    prefill(cache, e, a2, 600);                       // E_a2 supersedes E_a1 (B_a is already saved)
    CHECK(cache.slot_count() == 3);
    const auto sys_b = seq(20000, 600);
    const auto b1 = cat(sys_b, seq(25000, 100));
    prefill(cache, e, b1, 600);                       // B_b fits (4 slots), E_b1 forces one eviction: E_a1 (superseded)
    CHECK(cache.slot_count() == 4);
    CHECK(resume(cache, cat(a2, seq(7000, 10))) == a2.size());
    CHECK(resume(cache, cat(a1, seq(8000, 10))) == sys_a.size());  // E_a1 is gone, B_a serves it
    CHECK(resume(cache, cat(b1, seq(9000, 10))) == b1.size());
    // conversation c needs two slots: the end snapshots go (E_a2, least recently used, then E_b1) before any
    // boundary snapshot, so all three system prompts stay reusable
    const auto sys_c = seq(40000, 600);
    prefill(cache, e, cat(sys_c, seq(45000, 100)), 600);
    CHECK(cache.slot_count() == 4);
    CHECK(resume(cache, cat(a2, seq(7000, 10))) == sys_a.size());
    CHECK(resume(cache, cat(b1, seq(9000, 10))) == sys_b.size());
    CHECK(resume(cache, cat(sys_c, seq(47000, 10))) == sys_c.size());
}

void test_boundary_kept_for_dependents() {
    std::printf("a boundary snapshot is never evicted while an end snapshot needs its head rows\n");
    fake_engine e;
    PrefixSlotCache cache = make(e, 100000, 2);
    const auto sys = seq(1, 600);
    const auto t1 = cat(sys, seq(5000, 100));
    prefill(cache, e, t1, 600);                       // B (kv_from 0), E (kv_from 600)
    const auto t2 = cat(sys, seq(6000, 100));
    CHECK(resume(cache, t2) == sys.size());
    prefill(cache, e, t2, 600);                       // E2 needs B: E (not B) is evicted
    cache.clear_live();
    CHECK(resume(cache, cat(t2, seq(7000, 5))) == t2.size());
    CHECK(resume(cache, cat(sys, seq(8000, 5))) == sys.size());
}

void test_small_tail_not_saved() {
    std::printf("a side call a few tokens past the shared head is not saved, so it cannot push out the main conversation\n");
    fake_engine e;
    PrefixSlotCache cache = make(e, 100000, 2);
    const auto sys = seq(1, 600);
    const auto main1 = cat(sys, seq(5000, 1000));
    prefill(cache, e, main1, 600);                    // B, E_main
    for (int i = 0; i < 3; i++) {
        const auto side = cat(sys, seq(9000 + 100 * i, 17));
        CHECK(resume(cache, side) == sys.size());
        prefill(cache, e, side, 600);                 // 17 tokens past B: not saved
        CHECK(cache.slot_count() == 2);
        CHECK(resume(cache, cat(main1, seq(20000 + 100 * i, 30))) == main1.size());
    }
    // the first turn of a chat (a short user message after the head) is not saved either; the next turn resumes at the head
    const auto chat1 = cat(sys, seq(30000, 20));
    prefill(cache, e, chat1, 600);
    CHECK(resume(cache, cat(chat1, seq(31000, 30))) == sys.size());
}

void test_budget() {
    std::printf("the MiB budget evicts, and a prompt larger than the budget is not saved\n");
    fake_engine e;
    // 1 MiB state + 1 KiB per row; 3 MiB holds two 300-row self-contained snapshots, not three
    PrefixSlotCache cache = make(e, 3, 32);
    prefill(cache, e, seq(1, 300), 0);
    prefill(cache, e, seq(10000, 300), 0);
    prefill(cache, e, seq(20000, 300), 0);
    CHECK(cache.slot_count() == 2);
    CHECK(resume(cache, seq(1, 400)) == 0);           // the oldest went
    CHECK(resume(cache, seq(20000, 400)) == 300);
    // the default budget is 4096 MiB: a 2000 MiB snapshot fits with default flags (1024 MiB would refuse it)
    fake_engine huge;
    huge.state_bytes = 2000ull << 20;
    PrefixSlotCache defaults = make(huge);
    prefill(defaults, huge, seq(1, 10), 0);
    CHECK(defaults.slot_count() == 1);
    // 3000 rows = 1 MiB + 2.9 MiB > 3 MiB: refused
    fake_engine big;
    PrefixSlotCache small = make(big, 3, 32);
    prefill(small, big, seq(1, 3000), 0);
    CHECK(small.slot_count() == 0);
}

void test_images() {
    std::printf("a snapshot with images is only reused for the same pictures\n");
    fake_engine e;
    PrefixSlotCache cache = make(e);
    cache.image_token_id = 99;
    std::vector<int> prompt = cat(seq(1, 20), std::vector<int>(16, 99));
    prompt = cat(prompt, seq(200, 20));
    e.length = static_cast<int>(prompt.size());
    cache.set_live(prompt);
    cache.save(prompt, {111}, false, 0);
    const auto next = cat(prompt, seq(500, 10));
    CHECK(resume(cache, next, {222}) == 0);
    CHECK(resume(cache, next, {111}) == prompt.size());

    struct image { int grid_h, grid_w; };
    struct payload { std::vector<image> images; std::vector<uint16_t> _data__processed; };
    payload p1{{{2, 2}, {2, 2}}, std::vector<uint16_t>(2 * 2 * 4 * 3 * 2, 7)};
    payload p2 = p1;
    p2._data__processed.back() = 8;                   // second picture differs in one value
    const auto h1 = prefix_image_hashes(p1, 2, 1);
    const auto h2 = prefix_image_hashes(p2, 2, 1);
    CHECK(h1.size() == 2 && h2.size() == 2);
    CHECK(h1[0] == h2[0]);
    CHECK(h1[1] != h2[1]);
}

void test_restore_failure_and_off() {
    std::printf("a failed restore forgets every slot; 0 slots turns the cache off\n");
    fake_engine e;
    PrefixSlotCache cache = make(e);
    prefill(cache, e, seq(1, 600), 0);
    e.fail_restore = true;
    CHECK(resume(cache, seq(1, 700)) == 0);
    CHECK(cache.slot_count() == 0);
    CHECK(e.slots.empty());

    fake_engine off_e;
    PrefixSlotCache off = make(off_e, -1, 0);
    CHECK(!off.enabled());
    prefill(off, off_e, seq(1, 600), 0);
    CHECK(off.slot_count() == 0);
}
} // namespace

int main() {
    test_slot_zero_reserved();
    test_longest_strict_prefix();
    test_head_restored_after_cold_request();
    test_eviction_order();
    test_boundary_kept_for_dependents();
    test_small_tail_not_saved();
    test_budget();
    test_images();
    test_restore_failure_and_off();
    if (failures > 0) {
        std::printf("%d check(s) failed\n", failures);
        return 1;
    }
    std::printf("all passed\n");
    return 0;
}
