#include "models/phi4/rai/aie_next/phi4_rai_shape_plan.hpp"
#include "models/phi4/rai/aie_next/phi4_rai_constants.hpp"
#include "fake_corelib.hpp"
#include "test_support.hpp"

#include <algorithm>
#include <array>
#include <memory>
#include <string>

namespace {
using flm::corelib::CorelibApi;
using flm::phi4::Phi4ShapePlan;

std::shared_ptr<CorelibApi> Api() {
    return CorelibApi::ResolveForTest(fake_corelib::Resolver());
}

/// \brief a stream to build a plan against
/// \note Since corelib 0.5.0 every padding helper takes the stream,
///       because the PDI pair it was opened with selects the kernel set.
///       The fake hands back an opaque object; nothing dereferences it.
ryzenai_corelib_stream_ptr Stream(const std::shared_ptr<CorelibApi>& api) {
    void* raw = nullptr;
    api->Check(api->functions().create_stream(flm::phi4::kPrefillPdi,
                                              flm::phi4::kTokenPdi, &raw),
               "ryzenai_corelib_create_stream");
    return raw;
}

void TestShapePlanQueriesOnlyExecutionBucketsAndMapsEveryRow() {
    fake_corelib::Reset();
    const auto api = Api();
    const auto plan = Phi4ShapePlan::Build(api, Stream(api));
    const auto& counts = fake_corelib::GetState().call_counts;
    // 0.9 enumerates the shipped kernels once per op, then every bucket is
    // answered from that grid. The fake grid is a wildcard, so each requested
    // bucket is covered by itself and ForRows still snaps up to the next one.
    TEST_REQUIRE(counts.at("ryzenai_corelib_matmul_bf16_enum_kernels") == 1);
    TEST_REQUIRE(counts.at("ryzenai_corelib_ssmlp_bf16_enum_kernels") == 1);
    TEST_REQUIRE(counts.at("ryzenai_corelib_flat_mha_bf16_enum_kernels") == 1);
    TEST_REQUIRE(plan.ForRows(2).query_rows == 64);
    TEST_REQUIRE(plan.ForRows(65).query_rows == 128);
    TEST_REQUIRE(plan.ForRows(257).query_rows == 512);
    TEST_REQUIRE(plan.ForRows(4095).query_rows == 4096);
}

void TestShapePlanUsesExactQKvOutputSsmlpRmsAndLmHeadDimensions() {
    fake_corelib::Reset();
    const auto api = Api();
    const auto plan = Phi4ShapePlan::Build(api, Stream(api));
    TEST_REQUIRE(plan.lm_head_desc().k == 3072);
    TEST_REQUIRE(plan.lm_head_desc().n == 200064);
    TEST_REQUIRE(plan.lm_head_desc().group_size == 64);
    TEST_REQUIRE(plan.ForRows(1).query_rows == 1);
    TEST_REQUIRE(plan.ForRows(1).kv_rows == 1);
    TEST_REQUIRE(plan.ForRows(1).ssmlp_rows == 1);
    TEST_REQUIRE(plan.ForRows(1).flat_mha_rows == 1);
}

void TestShapePlanBuildsFlatMhaDescriptor24_8_128_4096_96() {
    fake_corelib::Reset();
    const auto api = Api();
    const auto plan = Phi4ShapePlan::Build(api, Stream(api));
    const auto& desc = plan.attention_desc();
    TEST_REQUIRE(desc.num_heads == 24);
    TEST_REQUIRE(desc.kv_num_heads == 8);
    TEST_REQUIRE(desc.head_size == 128);
    TEST_REQUIRE(desc.max_seq == 4096);
    TEST_REQUIRE(desc.rope_dim == 96);
}

void TestShapePlanRejectsAFailedKernelEnumeration() {
    fake_corelib::Reset();
    fake_corelib::GetState().statuses["ryzenai_corelib_matmul_bf16_enum_kernels"] =
        ryzenai_corelib_status_unsupported;
    RequireContains(RequireThrows([&] { const auto a = Api(); Phi4ShapePlan::Build(a, Stream(a)); }),
                    "ryzenai_corelib_matmul_bf16_enum_kernels");
}

void TestShapePlanRejectsRowsOutsideCachedRange() {
    fake_corelib::Reset();
    const auto api = Api();
    const auto plan = Phi4ShapePlan::Build(api, Stream(api));
    RequireContains(RequireThrows([&] { plan.ForRows(0); }), "1..4096");
    RequireContains(RequireThrows([&] { plan.ForRows(4097); }), "1..4096");
}

void TestShapePlanFailureNamesHelperAndLogicalShape() {
    fake_corelib::Reset();
    auto api = Api();
    fake_corelib::GetState().statuses["ryzenai_corelib_ssmlp_bf16_enum_kernels"] =
        ryzenai_corelib_status_unsupported;
    const auto error = RequireThrows([&] { Phi4ShapePlan::Build(api, Stream(api)); });
    RequireContains(error, "ryzenai_corelib_ssmlp_bf16_enum_kernels");
}
}  // namespace

int main() {
#define RUN_TEST(name) RunTest(&name, #name)
    RUN_TEST(TestShapePlanQueriesOnlyExecutionBucketsAndMapsEveryRow);
    RUN_TEST(TestShapePlanUsesExactQKvOutputSsmlpRmsAndLmHeadDimensions);
    RUN_TEST(TestShapePlanBuildsFlatMhaDescriptor24_8_128_4096_96);
    RUN_TEST(TestShapePlanRejectsAFailedKernelEnumeration);
    RUN_TEST(TestShapePlanRejectsRowsOutsideCachedRange);
    RUN_TEST(TestShapePlanFailureNamesHelperAndLogicalShape);
#undef RUN_TEST
}
