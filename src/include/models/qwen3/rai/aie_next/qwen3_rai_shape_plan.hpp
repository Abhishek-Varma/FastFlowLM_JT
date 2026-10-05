/// \file qwen3_rai_shape_plan.hpp
/// \brief The padded row count every op of one pass runs at
#pragma once

#include "models/qwen3/rai/aie_next/qwen3_rai_config.hpp"
#include "rai/corelib_api.hpp"

#include <cstddef>
#include <cstdint>
#include <memory>

namespace flm::qwen3 {

class Qwen3ShapePlan final {
public:
    /// \brief interrogate corelib at every execution bucket
    /// \throws std::runtime_error when any op pads a bucket to something else,
    ///         when a helper moves K or N, or when rows x heads lands in the
    ///         k=128 RMSNorm hole
    /// \note Qwen3 dispatches matmul, QK-Norm, flat_mha and ssmlp over the same
    ///       buffers, so they must agree on one row count; asking corelib, on
    ///       the stream the plan is for, turns a grid assumption into a load
    ///       failure rather than a wrong answer mid-generation.
    static Qwen3ShapePlan Build(const std::shared_ptr<const corelib::CorelibApi>& api,
                                ryzenai_corelib_stream_ptr stream,
                                const Qwen3Config& config);

    /// \brief the bucket a pass of `live_rows` runs at
    std::int64_t RowsFor(std::size_t live_rows) const;
    const ryzenai_corelib_flat_mha_bf16_desc& attention_desc() const noexcept { return attention_; }

private:
    ryzenai_corelib_flat_mha_bf16_desc attention_{};
};

}  // namespace flm::qwen3
