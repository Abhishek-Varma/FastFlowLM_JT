#include "models/qwen3/rai/aie_next/qwen3_rai_shape_plan.hpp"

#include "rai/kernel_grid.hpp"

#include <algorithm>
#include <stdexcept>
#include <string>

namespace flm::qwen3 {
namespace {

std::string Shape(std::int64_t m, std::int64_t k, std::int64_t n) {
    return "[" + std::to_string(m) + "," + std::to_string(k) + "]x[" + std::to_string(k) + "," +
           std::to_string(n) + "]";
}

void RequireMatmul(const corelib::ShapeGrid& grid, std::int64_t rows, std::int64_t k,
                   std::int64_t n, std::uint32_t group, const char* name) {
    const auto call = std::string(name) + " " + Shape(rows, k, n) + " g" + std::to_string(group);
    const auto covered = grid.SmallestM(rows, k, n, group);
    if (covered != rows)
        throw std::runtime_error(call + ": corelib does not ship this bucket exactly" +
                                 (covered < 0 ? "" : " (nearest " + std::to_string(covered) + ")"));
}

void RequireQkNormRows(std::int64_t rows, std::int64_t heads, const char* which) {
    const auto norm_rows = rows * heads;
    if (std::find(kQkNormRows.begin(), kQkNormRows.end(), norm_rows) == kQkNormRows.end())
        throw std::runtime_error(std::string(which) + "-norm would run " + std::to_string(rows) +
                                 " rows x " + std::to_string(heads) + " heads = " +
                                 std::to_string(norm_rows) +
                                 " rows at k=128, which no RMSNorm kernel ships");
}

}  // namespace

Qwen3ShapePlan Qwen3ShapePlan::Build(const std::shared_ptr<const corelib::CorelibApi>& api,
                                     ryzenai_corelib_stream_ptr stream,
                                     const Qwen3Config& c) {
    if (!api) throw std::invalid_argument("Qwen3ShapePlan corelib API is null");
    if (!stream) throw std::invalid_argument("Qwen3ShapePlan stream is null");
    if (c.head_dim != kQkNormWidth)
        throw std::runtime_error("Qwen3 head_dim " + std::to_string(c.head_dim) +
                                 " is not the width the QK-Norm grid was read at");

    Qwen3ShapePlan plan;
    // Full rotary: rope_dim == head_dim. A partial one selects a different
    // attention ELF rather than configuring the same one.
    plan.attention_ = {c.q_heads, c.kv_heads, c.head_dim, kMaxSequenceLength, c.head_dim};
    const auto matmul = corelib::ShapeGrid::Matmul(*api, stream);
    const auto ssmlp = corelib::ShapeGrid::SsMlp(*api, stream, false);
    const auto mha = corelib::ShapeGrid::FlatMha(*api, stream, plan.attention_);

    for (const auto rows : kExecutionRows) {
        RequireMatmul(matmul, rows, c.hidden, c.q_dim(), c.group, "query");
        RequireMatmul(matmul, rows, c.hidden, c.kv_dim(), c.group, "key/value");
        RequireMatmul(matmul, rows, c.q_dim(), c.hidden, c.group, "output");

        const auto ssmlp_rows = ssmlp.SmallestM(rows, c.hidden, c.intermediate, c.group);
        if (ssmlp_rows != rows)
            throw std::runtime_error("ssmlp [" + std::to_string(rows) + "," + std::to_string(c.hidden) +
                                     "," + std::to_string(c.intermediate) +
                                     "] does not ship exactly");

        const auto mha_rows = mha.SmallestM(rows, 0, 0, -1);
        if (mha_rows != rows)
            throw std::runtime_error("flat_mha [" + std::to_string(rows) +
                                     "] does not ship exactly");

        RequireQkNormRows(rows, c.q_heads, "q");
        RequireQkNormRows(rows, c.kv_heads, "k");
    }
    RequireMatmul(matmul, 1, c.hidden, c.vocab, c.head_group, "lm_head");
    return plan;
}

std::int64_t Qwen3ShapePlan::RowsFor(std::size_t live_rows) const {
    for (const auto rows : kExecutionRows)
        if (live_rows > 0 && static_cast<std::int64_t>(live_rows) <= rows) return rows;
    throw std::out_of_range("Qwen3 live rows must be in 1.." + std::to_string(kMaxSequenceLength));
}

}  // namespace flm::qwen3
