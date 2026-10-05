#include "models/gemma4/rai/aie_next/gemma4_rai_host.hpp"
#include "test_support.hpp"

#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <vector>

namespace {
using namespace flm::gemma4;

void TestRopeTableMatchesTheClosedForm() {
    const auto tables = flm::gemma4::MakeRopeTables(256, 1000000.0, 8);
    TEST_REQUIRE(tables.head_dim == 256);
    TEST_REQUIRE(tables.positions == 8);
    TEST_REQUIRE(tables.cos.size() == 8 * 128);
    for (std::int64_t pos = 0; pos < 8; ++pos) {
        for (std::int64_t i = 0; i < 128; ++i) {
            const double freq =
                1.0 / std::pow(1000000.0, (2.0 * static_cast<double>(i)) / 256.0);
            const double angle = static_cast<double>(pos) * freq;
            const auto index = static_cast<std::size_t>(pos * 128 + i);
            TEST_REQUIRE(std::abs(tables.cos[index] - std::cos(angle)) < 1e-5);
            TEST_REQUIRE(std::abs(tables.sin[index] - std::sin(angle)) < 1e-5);
        }
    }
}

void TestTheTwoThetasProduceDifferentTables() {
    const auto full = flm::gemma4::MakeRopeTables(512, 1000000.0, 4);
    const auto sliding = flm::gemma4::MakeRopeTables(512, 10000.0, 4);
    // Position 0 is all-ones/all-zeros for any theta; position 1 is where they
    // must already differ, which is the whole point of reading both keys.
    bool differs = false;
    for (std::size_t i = 256; i < 512; ++i) {
        if (std::abs(full.cos[i] - sliding.cos[i]) > 1e-4) { differs = true; break; }
    }
    TEST_REQUIRE(differs);
}

/// \brief `rope_freqs.weight` DIVIDES the frequency, and the direction is
///        pinned by arithmetic rather than by a magnitude
///
/// A uniform factor of 2 halves every frequency, so the factored table's angle
/// at position 2p is exactly the unfactored table's angle at position p. That
/// identity holds for every rotary pair at once and is what separates a
/// divisor from a multiplier: a multiplying implementation would put position
/// 2p's angle where position 4p's is. Neither is visible in any magnitude
/// assertion -- both tables are bounded by 1 and both look like rotary tables
/// -- which is why this is the shape of the check.
///
/// llama.cpp's own form is `theta /= freq_factors[i/2]`, applied on
/// FULL-ATTENTION LAYERS ONLY; the reference driver writes it as
/// `freq = freq / factors` (`gemma4_driver.py` `_rope_tables`).
void TestRopeFrequencyFactorsDivideTheFrequency() {
    const std::vector<float> halving(128, 2.0f);
    const auto plain = flm::gemma4::MakeRopeTables(256, 1000000.0, 8);
    const auto divided = flm::gemma4::MakeRopeTables(256, 1000000.0, 8, halving);
    for (std::size_t i = 0; i < 128; ++i) {
        // position 2 of the divided table == position 1 of the plain one
        TEST_REQUIRE(std::abs(divided.cos[2 * 128 + i] - plain.cos[1 * 128 + i]) < 1e-5f);
        TEST_REQUIRE(std::abs(divided.sin[2 * 128 + i] - plain.sin[1 * 128 + i]) < 1e-5f);
    }
    // And NOT position 4, which is where a multiplying implementation would
    // have put it. Checked at pair 0, whose frequency is exactly 1 and whose
    // angles are therefore far apart (1 radian against 4).
    TEST_REQUIRE(std::abs(divided.cos[2 * 128] - plain.cos[4 * 128]) > 1e-3f);

    // A factor vector of ones is the identity, so "applied" and "not applied"
    // cannot be told apart by accident.
    const std::vector<float> ones(128, 1.0f);
    const auto unity = flm::gemma4::MakeRopeTables(256, 1000000.0, 8, ones);
    for (std::size_t i = 0; i < unity.cos.size(); ++i)
        TEST_REQUIRE(std::abs(unity.cos[i] - plain.cos[i]) < 1e-6f);
}

/// \brief one factor per rotary PAIR, and a different length is refused
/// \note `rope_freqs.weight` is [256] on both shipped rows -- one entry per
///       pair of a 512-wide head, NOT a 256-wide rotary. A vector of the wrong
///       length would otherwise be read past its end or silently truncated.
void TestRopeFrequencyFactorsMustBeOnePerRotaryPair() {
    const std::vector<float> wrong(127, 2.0f);
    const auto error = RequireThrows([&] {
        (void)flm::gemma4::MakeRopeTables(256, 1000000.0, 4, wrong);
    });
    RequireContains(error, "127");
    RequireContains(error, "128");
}

void TestPositionZeroIsIdentity() {
    const auto tables = flm::gemma4::MakeRopeTables(256, 1000000.0, 2);
    for (std::size_t i = 0; i < 128; ++i) {
        TEST_REQUIRE(std::abs(tables.cos[i] - 1.0f) < 1e-6);
        TEST_REQUIRE(std::abs(tables.sin[i]) < 1e-6);
    }
}

void TestRmsNormAgainstTheClosedForm() {
    const std::vector<float> x{1.0f, -2.0f, 3.0f, -4.0f};
    const std::vector<float> gamma{0.5f, 1.0f, 1.5f, 2.0f};
    std::vector<float> out(4);
    flm::gemma4::RmsNorm(x, gamma, 1.0e-6f, out);
    double mean_square = 0.0;
    for (const float value : x) mean_square += double(value) * value;
    mean_square /= 4.0;
    const double scale = 1.0 / std::sqrt(mean_square + 1.0e-6);
    for (std::size_t i = 0; i < 4; ++i) {
        const auto expected = static_cast<float>(x[i] * scale * gamma[i]);
        TEST_REQUIRE(std::abs(out[i] - expected) < 1e-5f);
    }
}

void TestRmsNormRejectsAGammaOfADifferentWidth() {
    const std::vector<float> x(4, 1.0f);
    const std::vector<float> gamma(3, 1.0f);
    std::vector<float> out(4);
    const auto error = RequireThrows([&] {
        flm::gemma4::RmsNorm(x, gamma, 1.0e-6f, out);
    });
    RequireContains(error, "3");
    RequireContains(error, "4");
}

void TestDecodeQ8RowAppliesThePerBlockScale() {
    // One Q8_0 block: an fp16 scale of 0.5 then 32 int8 codes 0,1,2,...,31.
    std::vector<std::byte> row(2 + 32);
    const std::uint16_t half_scale = 0x3800;   // 0.5 in IEEE binary16
    std::memcpy(row.data(), &half_scale, 2);
    for (int i = 0; i < 32; ++i) {
        row[2 + std::size_t(i)] = static_cast<std::byte>(static_cast<std::int8_t>(i));
    }
    std::vector<float> out(32);
    flm::gemma4::DecodeQ8Row(row, out);
    for (int i = 0; i < 32; ++i) {
        TEST_REQUIRE(std::abs(out[std::size_t(i)] - float(i) * 0.5f) < 1e-4f);
    }
}

void TestFloatsToBf16RoundsToNearestEven() {
    // 1.0f is 0x3F800000; bf16 keeps the top 16 bits.
    const std::vector<float> values{1.0f, -1.0f, 0.0f};
    std::vector<std::uint16_t> out(3);
    flm::gemma4::FloatsToBf16(values, out);
    TEST_REQUIRE(out[0] == 0x3F80);
    TEST_REQUIRE(out[1] == 0xBF80);
    TEST_REQUIRE(out[2] == 0x0000);

    // The three cases above all have zero mantissa bits below the 16-bit cut,
    // so a plain truncating implementation ("out[i] = bits >> 16") passes them
    // identically to round-to-nearest-even. These three cases are built to
    // separate the two:
    //
    // 1) low 16 bits strictly above the halfway point (0x8001 > 0x8000) must
    //    round up; a truncating implementation stays down.
    //    bits = 0x3F808001 -> RNE adds (0x7FFF + bit0=0) = 0x7FFF, giving
    //    0x3F808001 + 0x00007FFF = 0x3F810000, upper half 0x3F81.
    //    Truncation gives 0x3F80. Verified by hand: 0x8001 + 0x7FFF = 0x10000,
    //    carries into the upper half.
    const float above_half = std::bit_cast<float>(std::uint32_t{0x3F808001});
    // 2) an exact tie (low 16 bits == 0x8000) whose upper half is ODD (bit0
    //    of 0x3F81 is 1) must round UP to the even neighbour 0x3F82; a
    //    truncating implementation would leave it at the odd 0x3F81.
    //    bits = 0x3F818000 -> RNE adds (0x7FFF + bit0=1) = 0x8000, giving
    //    0x3F818000 + 0x00008000 = 0x3F820000, upper half 0x3F82 (even).
    const float tie_odd_upper = std::bit_cast<float>(std::uint32_t{0x3F818000});
    // 3) an exact tie whose upper half is already EVEN (bit0 of 0x3F80 is 0)
    //    must be left ALONE at 0x3F80; this is the case a naive
    //    "always round half up" implementation gets wrong (it would produce
    //    0x3F81), even though it happens to agree with truncation here. This
    //    is what makes the test one of nearest-*even*, not round-half-up.
    //    bits = 0x3F808000 -> RNE adds (0x7FFF + bit0=0) = 0x7FFF, giving
    //    0x3F808000 + 0x00007FFF = 0x3F80FFFF, upper half 0x3F80 (unchanged).
    const float tie_even_upper = std::bit_cast<float>(std::uint32_t{0x3F808000});

    std::vector<float> rounding_values{above_half, tie_odd_upper, tie_even_upper};
    std::vector<std::uint16_t> rounding_out(3);
    flm::gemma4::FloatsToBf16(rounding_values, rounding_out);
    TEST_REQUIRE(rounding_out[0] == 0x3F81);
    TEST_REQUIRE(rounding_out[1] == 0x3F82);
    TEST_REQUIRE(rounding_out[2] == 0x3F80);
}

void TestSoftcapMatchesTanh() {
    std::vector<float> logits{0.0f, 1.0f, -1.0f, 100.0f, -100.0f};
    flm::gemma4::SoftcapLogits(logits, 30.0f);
    TEST_REQUIRE(std::abs(logits[0]) < 1e-6f);
    TEST_REQUIRE(std::abs(logits[1] - std::tanh(1.0f / 30.0f) * 30.0f) < 1e-5f);
    TEST_REQUIRE(std::abs(logits[2] - std::tanh(-1.0f / 30.0f) * 30.0f) < 1e-5f);
    // The cap is asymptotic: a big logit approaches 30 without reaching it.
    TEST_REQUIRE(logits[3] < 30.0f && logits[3] > 29.0f);
    TEST_REQUIRE(logits[4] > -30.0f && logits[4] < -29.0f);
}

void TestSoftcapOfZeroCapIsRejected() {
    std::vector<float> logits{1.0f};
    const auto error = RequireThrows([&] {
        flm::gemma4::SoftcapLogits(logits, 0.0f);
    });
    RequireContains(error, "cap");
}

// ONE TOKEN'S SLICE IS ONE CONTIGUOUS ROW, and this test was built the other
// way round until Task R1: it laid out `layers * ple_dim` rows of one block
// each and asserted the value picked out of each, which is exactly what the
// implementation did, so the two confirmed each other. On a real file
// `per_layer_token_embd.weight`'s dims are `[layers * ple_dim, vocab]` with
// dims[0] the FASTEST-VARYING extent -- so the bytes are `vocab` rows of
// `layers * ple_dim`, which is the transpose of what this used to build.
// Both layouts occupy the identical number of bytes, which is why nothing
// caught it; test_gemma4_real_gguf.cpp pins it against the reference driver
// over the real files, and this one pins the arithmetic.
void TestGatherPerLayerEmbeddingReadsOneContiguousRowPerToken() {
    // vocab 3 rows of (2 layers x ple_dim 32) = 64 elements = 2 Q8_0 blocks.
    constexpr std::int64_t kLayers = 2, kPleDim = 32, kVocab = 3;
    constexpr std::size_t kWidth = kLayers * kPleDim;      // 64
    constexpr std::size_t kBlocks = kWidth / 32;           // 2
    constexpr std::size_t kRowBytes = kBlocks * 34;        // 68
    std::vector<std::byte> tensor(static_cast<std::size_t>(kVocab) * kRowBytes);
    const std::uint16_t one = 0x3C00;   // 1.0 in binary16
    // Element e of token t is coded as t * 100 + e (mod int8), so a value
    // identifies BOTH which row was read and which position within it -- a
    // layout error that picked the right column out of the wrong rows, or
    // the right rows in the wrong order, shows up either way.
    for (std::size_t token = 0; token < kVocab; ++token) {
        for (std::size_t block = 0; block < kBlocks; ++block) {
            auto* base = tensor.data() + token * kRowBytes + block * 34;
            std::memcpy(base, &one, 2);
            for (std::size_t code = 0; code < 32; ++code) {
                const auto element = block * 32 + code;
                base[2 + code] = static_cast<std::byte>(
                    static_cast<std::int8_t>(token * 100 + element));
            }
        }
    }
    std::vector<float> out(kWidth);
    flm::gemma4::GatherPerLayerEmbedding(tensor, kLayers, kPleDim, kVocab, 1, out);
    TEST_REQUIRE(out.size() == 64);
    for (std::size_t element = 0; element < kWidth; ++element) {
        const auto expected =
            static_cast<float>(static_cast<std::int8_t>(100 + element));
        TEST_REQUIRE(std::abs(out[element] - expected) < 1e-4f);
    }
    // And a different token really does read a different row. The old
    // fixture had vocab 2 and only ever asked for token 1, so "which token"
    // was never varied at all.
    std::vector<float> other(kWidth);
    flm::gemma4::GatherPerLayerEmbedding(tensor, kLayers, kPleDim, kVocab, 2, other);
    for (std::size_t element = 0; element < kWidth; ++element) {
        const auto expected =
            static_cast<float>(static_cast<std::int8_t>(200 + element));
        TEST_REQUIRE(std::abs(other[element] - expected) < 1e-4f);
    }
}

// The PLE axis carries the Q8_0 blocks, so `layers * ple_dim` must divide by
// 32. Reading the tensor the other way round made this a constraint on the
// VOCABULARY instead, which is 262144 on both models and divides trivially --
// so the wrong rule was never exercised.
void TestGatherRejectsAWidthThatIsNotAWholeNumberOfBlocks() {
    std::vector<std::byte> tensor(34 * 4);
    std::vector<float> out(48);
    const auto error = RequireThrows([&] {
        flm::gemma4::GatherPerLayerEmbedding(tensor, 3, 16, 2, 0, out);
    });
    RequireContains(error, "48");
    RequireContains(error, "32");
}

void TestGatherRejectsATokenOutsideTheVocabulary() {
    std::vector<std::byte> tensor(34);
    std::vector<float> out(32);
    const auto error = RequireThrows([&] {
        flm::gemma4::GatherPerLayerEmbedding(tensor, 1, 32, 2, 7, out);
    });
    RequireContains(error, "7");
}

// A mapping shorter than `vocab` rows must be refused naming both lengths.
void TestGatherRejectsATruncatedMapping() {
    std::vector<std::byte> tensor(34);          // one row's worth, vocab 2
    std::vector<float> out(32);
    const auto error = RequireThrows([&] {
        flm::gemma4::GatherPerLayerEmbedding(tensor, 1, 32, 2, 1, out);
    });
    RequireContains(error, "34");
    RequireContains(error, "68");
}

// ---------------------------------------------------------------------------
// The per-layer embedding inputs -- the one thing a forward pass computes on
// the host.

/// \brief a Q8_0 table of `vocab` rows, each `width` elements, filled so that
///        element i of every row decodes to `0.5 * (i % 32)`
std::vector<std::byte> MakeQ8Table(std::int64_t vocab, std::int64_t width,
                                   bool zero_filled) {
    const auto blocks_per_row = static_cast<std::size_t>(width / 32);
    std::vector<std::byte> table(static_cast<std::size_t>(vocab) * blocks_per_row * 34,
                                 std::byte{0});
    if (zero_filled) return table;
    const std::uint16_t half_scale = 0x3800;   // 0.5 in IEEE binary16
    for (std::size_t block = 0; block < table.size() / 34; ++block) {
        std::memcpy(table.data() + block * 34, &half_scale, 2);
        for (int code = 0; code < 32; ++code)
            table[block * 34 + 2 + std::size_t(code)] =
                static_cast<std::byte>(static_cast<std::int8_t>(code));
    }
    return table;
}

/// \brief the per-layer embedding table's two scale factors, in isolation
///
/// WITH THE PROJECTION ZEROED, what is left is
/// `(0 + ple * sqrt(ple_dim)) * 1/sqrt(2)`, which pins both of the factors the
/// reference driver says are invisible if dropped: `sqrt(ple_dim)` on the
/// gathered per-layer embedding and `1/sqrt(2)` on the sum. Dropping either
/// produces finite, plausible, wrongly-scaled numbers -- the driver measured
/// `|out|.mean()` moving 0.636 -> 0.310 and 0.636 -> 0.899 respectively.
///
/// AND THE RESULT IS LAYER-MAJOR. `plane[l]` is `[rows, ple_dim]`, so a
/// layer's slice is one contiguous write into that layer's own device tensor.
void TestPerLayerInputsScaleTheEmbeddingAndHalveTheSum() {
    constexpr std::int64_t kLayers = 2, kPleDim = 32, kHidden = 32, kVocab = 2;
    const auto table = MakeQ8Table(kVocab, kLayers * kPleDim, /*zero_filled=*/false);
    const std::vector<std::uint16_t> model_proj(
        static_cast<std::size_t>(kLayers * kPleDim * kHidden), 0);
    const std::vector<float> proj_gamma(kPleDim, 1.0f);
    const std::vector<float> embedding(2 * kHidden, 3.0f);   // two rows
    const std::vector<int> ids{1, 0};

    const auto planes = flm::gemma4::BuildPerLayerInputs(
        table, model_proj, proj_gamma, embedding, ids, kLayers, kPleDim, kHidden,
        kVocab, 1.0e-6f);

    TEST_REQUIRE(planes.size() == static_cast<std::size_t>(kLayers));
    const float expected_scale =
        std::sqrt(static_cast<float>(kPleDim)) / std::sqrt(2.0f);
    for (std::size_t layer = 0; layer < planes.size(); ++layer) {
        TEST_REQUIRE(planes[layer].size() == 2 * kPleDim);
        for (std::size_t row = 0; row < 2; ++row) {
            for (std::size_t i = 0; i < kPleDim; ++i) {
                const float gathered = 0.5f * static_cast<float>(i);
                const auto value = planes[layer][row * kPleDim + i];
                TEST_REQUIRE(std::abs(value - gathered * expected_scale) < 1e-4f);
            }
        }
    }
}

/// \brief the projection reads `per_layer_model_proj` ROW-MAJOR, one row per
///        output column, and normalizes each layer's slice on its own
///
/// THE ORIENTATION IS THE SILENT ONE. `per_layer_model_proj.weight`'s file
/// dims are `[hidden, layers * ple_dim]` with the input width fastest-varying,
/// so the memory is `layers * ple_dim` rows of `hidden` -- and the matmul the
/// driver writes (`embed_rows(ids) @ floats(...)`) is against the TRANSPOSE of
/// that mapping, i.e. output column `c` is the dot product of the embedding
/// row with mapping row `c`. Reading it the other way round has the identical
/// element count and produces finite, plausible, wrong numbers.
///
/// Pinned with a projection whose row `c` is `(c + 1)` in its first column and
/// zero elsewhere, so output column `c` is `e[0] * (c + 1)` and a transposed
/// read cannot agree with it anywhere.
void TestPerLayerInputsProjectRowMajorAndNormalizePerLayer() {
    constexpr std::int64_t kLayers = 2, kPleDim = 32, kHidden = 32, kVocab = 2;
    const auto table = MakeQ8Table(kVocab, kLayers * kPleDim, /*zero_filled=*/true);
    std::vector<std::uint16_t> model_proj(
        static_cast<std::size_t>(kLayers * kPleDim * kHidden), 0);
    for (std::int64_t column = 0; column < kLayers * kPleDim; ++column) {
        const std::vector<float> value{static_cast<float>(column + 1)};
        std::vector<std::uint16_t> as_bf16(1);
        flm::gemma4::FloatsToBf16(value, as_bf16);
        model_proj[static_cast<std::size_t>(column * kHidden)] = as_bf16[0];
    }
    std::vector<float> proj_gamma(kPleDim);
    for (std::size_t i = 0; i < kPleDim; ++i) proj_gamma[i] = 0.25f + 0.05f * float(i);
    std::vector<float> embedding(kHidden, 0.0f);
    embedding[0] = 2.0f;
    const std::vector<int> ids{0};

    const auto planes = flm::gemma4::BuildPerLayerInputs(
        table, model_proj, proj_gamma, embedding, ids, kLayers, kPleDim, kHidden,
        kVocab, 1.0e-6f);

    TEST_REQUIRE(planes.size() == static_cast<std::size_t>(kLayers));
    for (std::int64_t layer = 0; layer < kLayers; ++layer) {
        // The raw projection for this layer's slice, before its norm. The
        // 1/sqrt(hidden) is applied to the projection, exactly as the driver
        // does -- and note that the RMSNorm below does NOT divide it back out,
        // because a uniform scale RESCALES EPSILON.
        std::vector<float> raw(kPleDim);
        for (std::int64_t i = 0; i < kPleDim; ++i) {
            const auto column = layer * kPleDim + i;
            raw[static_cast<std::size_t>(i)] =
                2.0f * static_cast<float>(column + 1) /
                std::sqrt(static_cast<float>(kHidden));
        }
        std::vector<float> normed(kPleDim);
        flm::gemma4::RmsNorm(raw, proj_gamma, 1.0e-6f, normed);
        for (std::int64_t i = 0; i < kPleDim; ++i) {
            const auto expected =
                normed[static_cast<std::size_t>(i)] / std::sqrt(2.0f);
            TEST_REQUIRE(std::abs(planes[static_cast<std::size_t>(layer)]
                                       [static_cast<std::size_t>(i)] -
                                  expected) < 1e-4f);
        }
    }
    // THE TWO LAYERS DIFFER. A reader that projected the whole width and then
    // normalized it in one go, rather than per layer slice, would produce the
    // same numbers for both -- and so would one that read the same slice
    // twice.
    TEST_REQUIRE(std::abs(planes[0][1] - planes[1][1]) > 1e-3f);
}

/// \brief the blocked, threaded projection agrees with a plain double dot
///        product at shapes that land on every edge of its tiling
///
/// `hidden` = 37 is not a multiple of the 8-wide SIMD step, 96 columns are two
/// column tiles (one of them partial), and the row counts leave 1, 2 and 3 rows
/// over from the 4-row tile and cross the 16-row block. The fixtures above are
/// 1-2 rows at hidden 32, which none of those paths ever see.
void TestPerLayerInputsProjectionMatchesAPlainDotProductAtOddShapes() {
    constexpr std::int64_t kLayers = 3, kPleDim = 32, kHidden = 37, kVocab = 2;
    constexpr std::int64_t kWidth = kLayers * kPleDim;
    const auto table = MakeQ8Table(kVocab, kWidth, /*zero_filled=*/true);
    std::uint32_t state = 12345;
    const auto next = [&] {
        state = state * 1664525u + 1013904223u;
        return static_cast<float>(state >> 8) / static_cast<float>(1u << 24) * 2.0f - 1.0f;
    };
    std::vector<float> weights_f(static_cast<std::size_t>(kWidth * kHidden));
    for (float& v : weights_f) v = next();
    std::vector<std::uint16_t> model_proj(weights_f.size());
    flm::gemma4::FloatsToBf16(weights_f, model_proj);
    const std::vector<float> proj_gamma(kPleDim, 1.0f);

    for (const std::size_t rows : {std::size_t{1}, std::size_t{18}, std::size_t{23}}) {
        std::vector<float> embedding(rows * kHidden);
        for (float& v : embedding) v = 2.0f * next();
        std::vector<int> ids(rows);
        for (std::size_t r = 0; r < rows; ++r) ids[r] = static_cast<int>(r % 2);

        const auto planes = flm::gemma4::BuildPerLayerInputs(
            table, model_proj, proj_gamma, embedding, ids, kLayers, kPleDim, kHidden,
            kVocab, 1.0e-6f);

        const float projection_scale = static_cast<float>(1.0 / std::sqrt(double(kHidden)));
        std::vector<float> projected(kWidth), normed(kPleDim);
        for (std::size_t r = 0; r < rows; ++r) {
            for (std::int64_t c = 0; c < kWidth; ++c) {
                double sum = 0.0;
                for (std::int64_t i = 0; i < kHidden; ++i) {
                    const auto bits = static_cast<std::uint32_t>(
                                          model_proj[std::size_t(c * kHidden + i)]) << 16;
                    sum += double(embedding[r * kHidden + std::size_t(i)]) *
                           double(std::bit_cast<float>(bits));
                }
                projected[std::size_t(c)] = static_cast<float>(sum) * projection_scale;
            }
            for (std::int64_t layer = 0; layer < kLayers; ++layer) {
                flm::gemma4::RmsNorm(
                    std::span<const float>(projected).subspan(std::size_t(layer * kPleDim),
                                                              kPleDim),
                    proj_gamma, 1.0e-6f, normed);
                for (std::int64_t i = 0; i < kPleDim; ++i) {
                    const float expected = normed[std::size_t(i)] / std::sqrt(2.0f);
                    const float actual =
                        planes[std::size_t(layer)][r * kPleDim + std::size_t(i)];
                    TEST_REQUIRE(std::abs(actual - expected) < 1e-6f);
                }
            }
        }
    }
}

}  // namespace

int main() {
#define RUN_TEST(name) RunTest(&name, #name)
    RUN_TEST(TestRopeTableMatchesTheClosedForm);
    RUN_TEST(TestTheTwoThetasProduceDifferentTables);
    RUN_TEST(TestRopeFrequencyFactorsDivideTheFrequency);
    RUN_TEST(TestRopeFrequencyFactorsMustBeOnePerRotaryPair);
    RUN_TEST(TestPositionZeroIsIdentity);
    RUN_TEST(TestRmsNormAgainstTheClosedForm);
    RUN_TEST(TestRmsNormRejectsAGammaOfADifferentWidth);
    RUN_TEST(TestDecodeQ8RowAppliesThePerBlockScale);
    RUN_TEST(TestFloatsToBf16RoundsToNearestEven);
    RUN_TEST(TestSoftcapMatchesTanh);
    RUN_TEST(TestSoftcapOfZeroCapIsRejected);
    RUN_TEST(TestGatherPerLayerEmbeddingReadsOneContiguousRowPerToken);
    RUN_TEST(TestGatherRejectsAWidthThatIsNotAWholeNumberOfBlocks);
    RUN_TEST(TestGatherRejectsATokenOutsideTheVocabulary);
    RUN_TEST(TestGatherRejectsATruncatedMapping);
    RUN_TEST(TestPerLayerInputsScaleTheEmbeddingAndHalveTheSum);
    RUN_TEST(TestPerLayerInputsProjectRowMajorAndNormalizePerLayer);
    RUN_TEST(TestPerLayerInputsProjectionMatchesAPlainDotProductAtOddShapes);
#undef RUN_TEST
}
