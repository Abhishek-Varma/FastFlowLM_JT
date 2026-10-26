/// \file prefix_slot_cache.hpp
/// \brief Token-level prefix cache over a few saved engine snapshots, shared by every model family that has them
/// \note An engine keeps up to N snapshots ("slots") of the state it cannot rewind (recurrent layers) plus the attention
///       K/V rows after a chosen position. This class remembers which tokens each slot was saved after, resumes a new
///       prompt from the longest slot that is a prefix of it, and decides what to throw away when memory or slots run out.
///       It does not know any model: the model wrapper hands it tokens and image hashes, and an adapter to its engine.
#pragma once
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

/// \brief what the cache needs from an engine; each engine class is adapted to it (see prefix_engine_adapter)
class prefix_engine_ops {
public:
    virtual ~prefix_engine_ops() = default;
    /// \brief save state, length and K/V rows from kv_from into a slot; returns the length saved, -1 on a bad slot
    virtual int checkpoint_slot(int slot, int kv_from) = 0;
    /// \brief go back to a slot (state, length, K/V rows from its kv_from, zero after); returns the length, -1 if empty
    virtual int restore_slot(int slot) = 0;
    /// \brief write back only the K/V rows a slot saved
    virtual int restore_slot_kv(int slot) = 0;
    virtual void drop_checkpoint_slot(int slot) = 0;
    /// \brief host bytes of one saved K/V row, and of one slot's recurrent state
    virtual size_t row_bytes() = 0;
    virtual size_t state_bytes() = 0;
    /// \brief how many slots the engine has
    virtual int max_slots() = 0;
};

/// \brief adapts an engine class that has the slot methods under the usual names
template <class Engine>
class prefix_engine_adapter final : public prefix_engine_ops {
public:
    explicit prefix_engine_adapter(Engine* engine) : engine_(engine) {}
    int checkpoint_slot(int slot, int kv_from) override { return engine_->checkpoint_slot(slot, kv_from); }
    int restore_slot(int slot) override { return engine_->restore_slot(slot); }
    int restore_slot_kv(int slot) override { return engine_->restore_slot_kv(slot); }
    void drop_checkpoint_slot(int slot) override { engine_->drop_checkpoint_slot(slot); }
    size_t row_bytes() override { return engine_->checkpoint_row_bytes(); }
    size_t state_bytes() override { return engine_->checkpoint_state_bytes(); }
    int max_slots() override { return Engine::NUM_CHECKPOINT_SLOTS; }

private:
    Engine* engine_;
};

/// \brief 64-bit content hash of a byte range, continuing from `seed`
uint64_t prefix_hash_bytes(const void* data, size_t bytes, uint64_t seed);

/// \brief content hash of each image of a payload, in order, so a snapshot is only reused for the same pictures
/// \note Works with any vision payload that has `images[i].grid_h / grid_w` and the concatenated pixel buffer
///       `_data__processed` (patch * patch * 3 * temporal values per grid cell).
template <class Payload>
std::vector<uint64_t> prefix_image_hashes(const Payload& payload, size_t patch, size_t temporal) {
    std::vector<uint64_t> hashes;
    const auto& pixels = payload._data__processed;
    size_t offset = 0;
    for (const auto& img : payload.images) {
        const size_t count = static_cast<size_t>(img.grid_h) * patch * static_cast<size_t>(img.grid_w) * patch * 3u * temporal;
        uint64_t h = 0xcbf29ce484222325ull ^ (static_cast<uint64_t>(img.grid_h) << 32 | static_cast<uint64_t>(img.grid_w));
        if (offset + count <= pixels.size()) {
            h = prefix_hash_bytes(pixels.data() + offset, count * sizeof(pixels[0]), h);
        }
        hashes.push_back(h);
        offset += count;
    }
    return hashes;
}

class PrefixSlotCache {
public:
    /// \brief engine slot 0 is the model's single checkpoint()/restore(); the cache hands out 1 and up only
    static constexpr int kFirstSlot = 1;

    /// \brief the placeholder token id of images; slots count the image runs they contain
    int image_token_id = -1;
    /// \brief a shared head shorter than this is not worth a slot: re-processing it is cheap
    size_t min_boundary = 512;
    /// \brief an end snapshot must add at least this many tokens over a saved slot that covers its prompt
    size_t min_end_gain = 64;

    /// \brief --prefix-cache-mb / --prefix-slots (-1 = default; 0 slots = off)
    void configure(int cache_mb, int slots) {
        cache_mb_ = cache_mb;
        slots_ = slots;
    }
    /// \brief attach the engine (nullptr detaches); slots of a former engine are forgotten
    void attach(std::unique_ptr<prefix_engine_ops> ops);
    bool enabled() const { return ops_ != nullptr && slots_ != 0; }

    /// \brief result of resume()
    struct resume_t {
        bool resumed = false;      ///< the engine now holds a slot's state
        size_t match_len = 0;      ///< tokens of the prompt it covers
        int slot_id = -1;
    };
    /// \brief resume from the longest saved prefix of `full_tokens` and restore the engine to it
    /// \param full_tokens what the prompt leaves in the engine
    /// \param image_hashes content hash of each image of the prompt, in order
    /// \param matched_tokens set to the tokens of the slot used (what the engine now holds)
    /// \note Only slots strictly shorter than the prompt count: one token must still be processed to get logits.
    /// \note On a restore failure every slot is forgotten and resumed is false.
    resume_t resume(const std::vector<int>& full_tokens, const std::vector<uint64_t>& image_hashes, std::vector<int>& matched_tokens);

    /// \brief take a slot at the engine's current position; `tokens` is what the engine holds
    /// \param boundary true for the end of system + tools (kv_from 0, evicted last)
    /// \param kv_from first position whose K/V rows the slot saves (0 = self-contained)
    void save(const std::vector<int>& tokens, const std::vector<uint64_t>& image_hashes, bool boundary, int kv_from);

    /// \brief the kv_from an end-of-prompt slot should use: `boundary` when a self-contained slot holds the first
    ///        `boundary` tokens of `tokens`, else 0
    int end_kv_from(const std::vector<int>& tokens, size_t boundary) const;

    /// \brief tokens whose K/V rows the engine's cache holds now (what the last request left in it)
    void set_live(const std::vector<int>& tokens) { live_tokens_ = tokens; }
    void clear_live() { live_tokens_.clear(); }

    /// \brief forget every slot (and the live tokens)
    void forget_all();

    size_t slot_count() const { return slots_table_.size(); }

private:
    struct slot_t {
        int id = 0;                          // engine slot
        std::vector<int> tokens;             // what the slot was saved after
        std::vector<uint64_t> image_hashes;  // content of the images inside `tokens`
        int kv_from = 0;                     // first position whose K/V rows the slot holds
        uint64_t last_used = 0;
        bool boundary = false;               // end of system + tools; evicted last
    };

    size_t budget_bytes() const;
    size_t slot_limit() const;
    size_t count_image_runs(const std::vector<int>& tokens, size_t len) const;

    std::unique_ptr<prefix_engine_ops> ops_;
    std::vector<slot_t> slots_table_;
    std::vector<int> live_tokens_;
    uint64_t clock_ = 0;
    int cache_mb_ = -1;
    int slots_ = -1;
    size_t row_bytes_ = 0;
    size_t state_bytes_ = 0;
    bool announced_ = false;
    bool warned_too_big_ = false;
};
