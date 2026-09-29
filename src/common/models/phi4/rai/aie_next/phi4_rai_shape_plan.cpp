#include "models/phi4/rai/aie_next/phi4_rai_shape_plan.hpp"

#include "models/phi4/rai/aie_next/phi4_rai_constants.hpp"
#include "rai/kernel_grid.hpp"

#include <algorithm>
#include <array>
#include <stdexcept>
#include <string>

namespace flm::phi4 {
namespace {

std::int64_t MatmulRows(const corelib::ShapeGrid& grid, std::int64_t rows,
                        std::int64_t logical_k, std::int64_t logical_n,
                        const char* logical_name) {
    return corelib::CoveringRows(
        grid, rows, logical_k, logical_n, kRequantizedGroupSize, logical_name);
}

}  // namespace

Phi4ShapePlan Phi4ShapePlan::Build(
    const std::shared_ptr<const corelib::CorelibApi>& api,
    ryzenai_corelib_stream_ptr stream) {
    if (!api) throw std::invalid_argument("Phi4ShapePlan corelib API is null");
    if (!stream) throw std::invalid_argument("Phi4ShapePlan stream is null");

    Phi4ShapePlan plan;
    plan.attention_desc_ = {kQueryHeadCount, kKvHeadCount, kHeadSize,
                            kMaxSequenceLength, kRopeDimension};
    plan.lm_head_desc_ = {kHiddenSize, kVocabularySize,
                          kRequantizedGroupSize, false};
    const auto matmul = corelib::ShapeGrid::Matmul(*api, stream);
    const auto ssmlp = corelib::ShapeGrid::SsMlp(*api, stream, false);
    const auto mha = corelib::ShapeGrid::FlatMha(*api, stream, plan.attention_desc_);
    plan.rows_.reserve(kMaxSequenceLength);
    constexpr std::array<std::int64_t, 8> execution_rows{
        1, 64, 128, 256, 512, 1024, 2048, 4096};

    for (const auto rows : execution_rows) {
        Phi4RowExtents extents{};
        extents.query_rows = MatmulRows(matmul, rows, kHiddenSize,
                                        kQueryDimension, "query");
        extents.kv_rows = MatmulRows(matmul, rows, kHiddenSize,
                                     kKvDimension, "key/value");
        extents.output_rows = MatmulRows(matmul, rows, kHiddenSize,
                                         kHiddenSize, "output");
        extents.ssmlp_rows = corelib::CoveringRows(
            ssmlp, rows, kHiddenSize, kIntermediateSize, kRequantizedGroupSize, "ssmlp");
        extents.flat_mha_rows = corelib::CoveringRows(mha, rows, 0, 0, -1, "flat_mha");
        plan.maximum_extents_.query_rows = std::max(
            plan.maximum_extents_.query_rows, extents.query_rows);
        plan.maximum_extents_.kv_rows = std::max(
            plan.maximum_extents_.kv_rows, extents.kv_rows);
        plan.maximum_extents_.output_rows = std::max(
            plan.maximum_extents_.output_rows, extents.output_rows);
        plan.maximum_extents_.ssmlp_rows = std::max(
            plan.maximum_extents_.ssmlp_rows, extents.ssmlp_rows);
        plan.maximum_extents_.flat_mha_rows = std::max(
            plan.maximum_extents_.flat_mha_rows, extents.flat_mha_rows);
        while (plan.rows_.size() < static_cast<std::size_t>(rows))
            plan.rows_.push_back(extents);
    }

    (void)MatmulRows(matmul, 1, kHiddenSize, kVocabularySize, "lm_head");
    return plan;
}

const Phi4RowExtents& Phi4ShapePlan::ForRows(std::size_t live_rows) const {
    if (live_rows == 0 || live_rows > rows_.size()) {
        throw std::out_of_range("Phi-4 live rows must be in 1..4096");
    }
    return rows_[live_rows - 1];
}

const Phi4RowExtents& Phi4ShapePlan::maximum_extents() const noexcept {
    return maximum_extents_;
}

const ryzenai_corelib_flat_mha_bf16_desc&
Phi4ShapePlan::attention_desc() const noexcept {
    return attention_desc_;
}

const ryzenai_corelib_matmul_bf16_weights_desc&
Phi4ShapePlan::lm_head_desc() const noexcept {
    return lm_head_desc_;
}

}  // namespace flm::phi4
