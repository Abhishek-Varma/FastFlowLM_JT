/// \file d3_engine_adapter.cpp
/// \brief See d3_engine_adapter.hpp.
#include "AutoDecisionModel/d3_engine_adapter.hpp"

#include <cstring>
#include <map>
#include <stdexcept>

#include "models/d3/flm/aie2p/d3_npu.hpp"   // d3_image_payload

namespace {

using flm::d3::EngineBlock;
using flm::d3::EngineImage;
using flm::d3::EngineStateRequest;

class D3EngineAdapter : public flm::d3::ReadoutEngine {
public:
    D3EngineAdapter(decision_model* engine, int image_token_id) : engine_(engine), image_token_id_(image_token_id) {}

    int hidden_size() const override { return engine_->hidden_size(); }

    std::vector<std::vector<float>> encode_state_once(const EngineStateRequest& request) override {
        // M1 path: every block is a prompt [state | block] with state_len = |state|; one hidden row (the last
        // token) comes back per prompt. Blocks that read other positions need the engine's multi-position output.
        for (const EngineBlock& b : request.blocks) {
            if (b.readout_from_end != std::vector<int>{0}) {
                throw std::runtime_error(
                    "this d3 engine returns the last hidden row of a prompt only; a Box question reads four "
                    "(the engine's multi-position output, encode_state_once, is not in this build yet)");
            }
        }
        // prompts that carry the same images share one batch: the engine's payload lists the images of the whole batch
        std::map<std::vector<int>, std::vector<size_t>> by_images;
        for (size_t b = 0; b < request.blocks.size(); b++) by_images[request.blocks[b].image_ids].push_back(b);
        std::vector<std::vector<float>> out(request.blocks.size());
        const int hidden = engine_->hidden_size();
        for (const auto& group : by_images) {
            d3_image_payload payload;
            payload.image_token_id = request.image_token_id >= 0 ? request.image_token_id : image_token_id_;
            std::vector<int> order = request.state_image_ids;
            order.insert(order.end(), group.first.begin(), group.first.end());
            for (int idx : order) {
                const EngineImage& im = request.images.at(idx);
                payload.images.push_back({im.grid_h, im.grid_w});
                const size_t at = payload.pixels.size();
                payload.pixels.resize(at + im.pixels.size());
                static_assert(sizeof(bf16) == sizeof(uint16_t), "bf16 is 16 bits");
                std::memcpy(payload.pixels.data() + at, im.pixels.data(), im.pixels.size() * sizeof(uint16_t));
            }
            std::vector<std::vector<int>> ids;
            for (size_t b : group.second) {
                std::vector<int> row = request.state_ids;
                row.insert(row.end(), request.blocks[b].ids.begin(), request.blocks[b].ids.end());
                ids.push_back(std::move(row));
                payload.state_len.push_back(static_cast<int>(request.state_ids.size()));
            }
            // moved, never copied: XRT buffer copies do not own their memory
            std::vector<buffer<bf16>> rows = engine_->encode_batch(ids, &payload);
            for (size_t i = 0; i < group.second.size(); i++) {
                std::vector<float> h(hidden);
                const bf16* row = rows[i].data();
                for (int d = 0; d < hidden; d++) h[d] = static_cast<float>(row[d]);
                out[group.second[i]] = std::move(h);
            }
        }
        return out;
    }

private:
    decision_model* engine_;
    int image_token_id_;
};

}  // namespace

std::unique_ptr<flm::d3::ReadoutEngine> make_d3_engine_adapter(decision_model* engine, int image_token_id_hint) {
    return std::make_unique<D3EngineAdapter>(engine, image_token_id_hint);
}
