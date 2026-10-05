#include "models/gemma4/rai/aie_next/gemma4_rai_gguf.hpp"
#include "gemma4_gguf_fixture.hpp"
#include "rai/gguf_file.hpp"
#include "test_support.hpp"

#include <algorithm>
#include <cstdint>
#include <utility>
#include <vector>

namespace {
using flm::rai::GgufFile;
using flm::gemma4::DeriveCacheOwners;
using flm::gemma4::DeriveFfnWidths;
using flm::gemma4::DeriveLayerIsSwa;

void TestE2bFfnWidthStepsAtLayer15DespiteTheArrayMetadata() {
    auto file = GgufFile::Open(gemma4_fixture::Write(gemma4_fixture::E2bOptions()));
    const auto widths = DeriveFfnWidths(*file, 35);
    TEST_REQUIRE(widths.size() == 35);
    TEST_REQUIRE(widths[0] == 6144);
    TEST_REQUIRE(widths[14] == 6144);
    TEST_REQUIRE(widths[15] == 12288);
    TEST_REQUIRE(widths[34] == 12288);
}

void TestE4bFfnWidthIsUniformDespiteTheScalarMetadata() {
    auto file = GgufFile::Open(gemma4_fixture::Write(gemma4_fixture::E4bOptions()));
    const auto widths = DeriveFfnWidths(*file, 42);
    TEST_REQUIRE(widths.size() == 42);
    for (const auto width : widths) {
        TEST_REQUIRE(width == 10240);
    }
}

// The one that matters: write E2B's real tensors (stepping 6144 -> 12288 at
// layer 15), then overwrite gemma4.feed_forward_length with a 35-entry array
// that LIES -- every entry 999 -- using the same Builder-before-Write() seam
// test_gemma4_config.cpp's
// TestFeedForwardLengthMetadataIsSettableIndependentlyOfTensorShapes uses.
// MakeBuilder has already committed every tensor's shape before this runs, so
// overwriting the metadata key cannot touch them. If the derivation still
// reports the true, stepping widths, it never read the key at all.
void TestFfnWidthIgnoresTheMetadataKeyEntirely() {
    const auto options = gemma4_fixture::E2bOptions();
    auto builder = gemma4_fixture::MakeBuilder(options);
    const auto tensor_count_before = builder.TensorCount();

    std::vector<std::byte> lie;
    for (int i = 0; i < 35; ++i) {
        gemma4_fixture::Append(lie, std::uint32_t{999});
    }
    builder.SetMetadata("gemma4.feed_forward_length",
                        gemma4_fixture::ArrayValue{gemma4_fixture::kMetaUint32,
                                                   35, std::move(lie)});
    TEST_REQUIRE(builder.TensorCount() == tensor_count_before);   // no tensor touched

    auto file = GgufFile::Open(builder.Write());
    // Confirm the metadata really does say the lie, so the assertions below
    // are proof the derivation ignored it -- not merely that no one changed it.
    const auto lying_metadata = file->UnsignedArray("gemma4.feed_forward_length");
    TEST_REQUIRE(lying_metadata[0] == 999);
    TEST_REQUIRE(lying_metadata[15] == 999);

    const auto widths = DeriveFfnWidths(*file, 35);
    TEST_REQUIRE(widths[0] == 6144);
    TEST_REQUIRE(widths[14] == 6144);
    TEST_REQUIRE(widths[15] == 12288);
    TEST_REQUIRE(widths[34] == 12288);
}

void TestRejectsNonTwoDGateProjection() {
    gemma4_fixture::Builder builder;
    builder.AddTensor("blk.0.ffn_gate.weight", {6144}, gemma4_fixture::kF32);
    auto file = GgufFile::Open(builder.Write());
    const auto error = RequireThrows([&] {
        DeriveFfnWidths(*file, 1);
    });
    RequireContains(error, "blk.0.ffn_gate.weight");
}

// Confirms DeriveLayerIsSwa's answer against E2B's real full-attention set
// -- {4, 9, 14, 19, 24, 29, 34}, period 5 -- WITHOUT the derivation itself
// ever computing a period: gemma4_gguf_fixture.hpp writes attn_q's width
// from the layer's own head_dim (2048 sliding / 4096 full), and this test
// only checks that DeriveLayerIsSwa reproduces the resulting set. See
// TestE4bFullAttentionLayersAreEverySixth: the same derivation function
// reproduces a DIFFERENT set from a DIFFERENT period below, which a
// period-based implementation could not do for both.
void TestE2bFullAttentionLayersAreEveryFifth() {
    auto file = GgufFile::Open(gemma4_fixture::Write(gemma4_fixture::E2bOptions()));
    const auto is_swa = DeriveLayerIsSwa(*file, 35, 8, 256, 512);
    TEST_REQUIRE(is_swa.size() == 35);
    const std::vector<int> expected_full{4, 9, 14, 19, 24, 29, 34};
    for (std::size_t layer = 0; layer < is_swa.size(); ++layer) {
        const bool should_be_full =
            std::find(expected_full.begin(), expected_full.end(),
                      static_cast<int>(layer)) != expected_full.end();
        TEST_REQUIRE(is_swa[layer] == !should_be_full);
    }
}

// E4B's full-attention set -- {5, 11, 17, 23, 29, 35, 41}, period 6 -- is a
// different period from E2B's above. The same DeriveLayerIsSwa call, with
// no knowledge of either period, gets both right because it reads each
// layer's own attn_q width instead of tabulating either model's cadence.
void TestE4bFullAttentionLayersAreEverySixth() {
    auto file = GgufFile::Open(gemma4_fixture::Write(gemma4_fixture::E4bOptions()));
    const auto is_swa = DeriveLayerIsSwa(*file, 42, 8, 256, 512);
    TEST_REQUIRE(is_swa.size() == 42);
    const std::vector<int> expected_full{5, 11, 17, 23, 29, 35, 41};
    for (std::size_t layer = 0; layer < is_swa.size(); ++layer) {
        const bool should_be_full =
            std::find(expected_full.begin(), expected_full.end(),
                      static_cast<int>(layer)) != expected_full.end();
        TEST_REQUIRE(is_swa[layer] == !should_be_full);
    }
}

// The fixture's FixtureOptions has no knob for a corrupted attn_q width (and
// this task does not own gemma4_gguf_fixture.hpp to add one -- an agent is
// mid-fix on that file's find_path scoping and temp-filename collision).
// gemma4_fixture::Builder is exposed for exactly this, though, and
// TestRejectsNonTwoDGateProjection above already uses it the same way: build
// a minimal file by hand rather than through MakeBuilder/FixtureOptions.
// Only blk.N.attn_q.weight is read by DeriveLayerIsSwa, so only that tensor,
// for layers 0-3, needs to exist. Layer 3's width (1234) is neither
// sliding's 2048 nor full's 4096 at q_heads=8/head_dim=256/global_head_dim=512.
//
// Shapes are in `AddTensor`'s (== `ShapeOf`'s) [OUT, IN] convention, so the
// OUTPUT width comes FIRST -- see gemma4_gguf_fixture.hpp. Written the other
// way round, as this test was until Task R1, the "corrupt" layer would be
// 1536 wide like every other and the derivation would reject layer 0 first,
// so the test would pass while proving nothing about layer 3.
void TestAttnQWidthMatchingNeitherGeometryIsRejected() {
    gemma4_fixture::Builder builder;
    builder.AddTensor("blk.0.attn_q.weight", {2048, 1536}, gemma4_fixture::kQ8_0);
    builder.AddTensor("blk.1.attn_q.weight", {2048, 1536}, gemma4_fixture::kQ8_0);
    builder.AddTensor("blk.2.attn_q.weight", {2048, 1536}, gemma4_fixture::kQ8_0);
    builder.AddTensor("blk.3.attn_q.weight", {1234, 1536}, gemma4_fixture::kQ8_0);
    auto file = GgufFile::Open(builder.Write());
    const auto error = RequireThrows([&] {
        DeriveLayerIsSwa(*file, 4, 8, 256, 512);
    });
    RequireContains(error, "blk.3.attn_q.weight");
}

// llama.cpp's rule: a shared layer reads the LAST OWN-CACHE LAYER OF ITS OWN
// KIND, scanning only the owning prefix (the first `kv_layers` layers, each
// of which maps to itself). 15 owning on E2B gives 13 sliding / 14 full;
// see TestTheTwoModelsDisagreeAboutTheSharedCaches below for why this
// specific pair cannot be guessed from E4B's.
void TestE2bSharedLayersReadCaches13And14() {
    auto file = GgufFile::Open(gemma4_fixture::Write(gemma4_fixture::E2bOptions()));
    const auto is_swa = DeriveLayerIsSwa(*file, 35, 8, 256, 512);
    const auto owners = DeriveCacheOwners(is_swa, 15);
    TEST_REQUIRE(owners.size() == 35);
    for (std::int64_t layer = 0; layer < 15; ++layer) {
        TEST_REQUIRE(owners[static_cast<std::size_t>(layer)] == layer);   // owns its own
    }
    for (std::size_t layer = 15; layer < owners.size(); ++layer) {
        TEST_REQUIRE(owners[layer] == (is_swa[layer] ? 13 : 14));
    }
}

// Same rule, 24 owning on E4B, gives 22 sliding / 23 full -- a DIFFERENT pair
// from E2B's 13/14, off a different owning-prefix size and a different
// full-attention period. Neither pair is derivable from the other.
void TestE4bSharedLayersReadCaches22And23() {
    auto file = GgufFile::Open(gemma4_fixture::Write(gemma4_fixture::E4bOptions()));
    const auto is_swa = DeriveLayerIsSwa(*file, 42, 8, 256, 512);
    const auto owners = DeriveCacheOwners(is_swa, 24);
    TEST_REQUIRE(owners.size() == 42);
    for (std::int64_t layer = 0; layer < 24; ++layer) {
        TEST_REQUIRE(owners[static_cast<std::size_t>(layer)] == layer);
    }
    for (std::size_t layer = 24; layer < owners.size(); ++layer) {
        TEST_REQUIRE(owners[layer] == (is_swa[layer] ? 22 : 23));
    }
}

// The point of the two tests above, made explicit as one assertion: a
// constant lifted from either model's shared-cache pair is wrong about the
// other. This is what makes DeriveCacheOwners a derivation and not a table.
void TestTheTwoModelsDisagreeAboutTheSharedCaches() {
    auto e2b = GgufFile::Open(gemma4_fixture::Write(gemma4_fixture::E2bOptions()));
    auto e4b = GgufFile::Open(gemma4_fixture::Write(gemma4_fixture::E4bOptions()));
    const auto e2b_owners =
        DeriveCacheOwners(DeriveLayerIsSwa(*e2b, 35, 8, 256, 512), 15);
    const auto e4b_owners =
        DeriveCacheOwners(DeriveLayerIsSwa(*e4b, 42, 8, 256, 512), 24);
    TEST_REQUIRE(e2b_owners.back() != e4b_owners.back());

    // The check above is not enough on its own: layer 34 (E2B) and layer 41
    // (E4B) -- the LAST layer of each model -- are both full-attention, and
    // their owners (14 and 23) happen to equal kv_layers - 1 in BOTH models
    // (15-1=14, 24-1=23). A kind-blind bug that maps EVERY shared layer to
    // kv_layers - 1, without ever checking whether the layer is sliding or
    // full, would reproduce 14 and 23 here too, and the assertion above
    // would still pass against it.
    //
    // Layer 33 (E2B) and layer 40 (E4B) are both SLIDING and shared. The
    // real, kind-aware algorithm maps them to the last SLIDING owner (13
    // and 22 respectively) -- which differs from kv_layers - 1 (14 and 23)
    // in both models. A kind-blind bug would instead map them to
    // kv_layers - 1, i.e. 14 and 23, failing these assertions.
    TEST_REQUIRE(e2b_owners[33] == 13);
    TEST_REQUIRE(e4b_owners[40] == 22);
    TEST_REQUIRE(e2b_owners[33] != e4b_owners[40]);
}

// kv_layers means "how many layers OWN a cache" -- it must land inside
// 1..layers or every layer past the end of the vector is undefined. This is
// deliberately hand-built rather than fixture-derived: the fixture writes
// the FILE's shared_kv_layers semantics (20/18), never the owning count this
// function takes, so a bogus kv_layers has nothing to do with either
// fixture's real numbers.
void TestRejectsKvLayersOutsideRange() {
    const std::vector<bool> is_swa{true, false, true, false};
    {
        const auto error = RequireThrows([&] { DeriveCacheOwners(is_swa, 0); });
        RequireContains(error, "0");
    }
    {
        const auto error = RequireThrows([&] { DeriveCacheOwners(is_swa, 5); });
        RequireContains(error, "5");
    }
}

// A shared layer whose kind (sliding or full-attention) has no owning layer
// at all cannot pick a cache to read -- the owning prefix here is all
// full-attention (layers 0-1), so layer 2's sliding share has nothing to
// bind to. The error must name both the layer and the kind, the same way
// TestAttnQWidthMatchingNeitherGeometryIsRejected names the tensor above.
void TestRejectsShareOfAKindNoOwningLayerHas() {
    const std::vector<bool> is_swa{false, false, true};
    const auto error = RequireThrows([&] { DeriveCacheOwners(is_swa, 2); });
    RequireContains(error, "layer 2");
    RequireContains(error, "sliding");
}
}  // namespace

int main() {
#define RUN_TEST(name) RunTest(&name, #name)
    RUN_TEST(TestE2bFfnWidthStepsAtLayer15DespiteTheArrayMetadata);
    RUN_TEST(TestE4bFfnWidthIsUniformDespiteTheScalarMetadata);
    RUN_TEST(TestFfnWidthIgnoresTheMetadataKeyEntirely);
    RUN_TEST(TestRejectsNonTwoDGateProjection);
    RUN_TEST(TestE2bFullAttentionLayersAreEveryFifth);
    RUN_TEST(TestE4bFullAttentionLayersAreEverySixth);
    RUN_TEST(TestAttnQWidthMatchingNeitherGeometryIsRejected);
    RUN_TEST(TestE2bSharedLayersReadCaches13And14);
    RUN_TEST(TestE4bSharedLayersReadCaches22And23);
    RUN_TEST(TestTheTwoModelsDisagreeAboutTheSharedCaches);
    RUN_TEST(TestRejectsKvLayersOutsideRange);
    RUN_TEST(TestRejectsShareOfAKindNoOwningLayerHas);
#undef RUN_TEST
}
