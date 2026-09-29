// THE ONLY TEST IN THIS SUITE THAT OPENS A FILE GOOGLE SHIPPED.
//
// Every other Gemma 4 test reads `gemma4_gguf_fixture.hpp`'s synthetic file.
// That fixture was written from verified-gguf-facts.md's tables, and three
// readers were then written to match the fixture, so fixture and readers
// confirmed each other through four signed-off tasks while being transposed
// relative to every real conversion. A fixture that confirms itself passes
// unconditionally and is worse than no test, because it looks like coverage.
//
// This binary is the answer to that: it asserts against numbers HAND-WRITTEN
// from verified-gguf-facts.md and never derived from the file under test, and
// it runs them over `gemma-4-E2B-it-Q8_0.gguf` and `gemma-4-E4B-it-Q8_0.gguf`
// themselves. If the fixture and the readers ever agree with each other and
// disagree with Google, this is what says so.
//
// OPT-IN, BECAUSE THE FILES ARE 4.8 GB AND 7.7 GB and will not exist on every
// machine. The two environment variables below name them. The skip is LOUD:
// with NEITHER set the process returns 77 and ctest prints "Skipped" on its
// own line, and with exactly ONE set it FAILS rather than running half the
// assertions and reporting green -- a run that covers less than it appears to
// is the failure this project keeps paying for.
//
//   set FLM_GEMMA4_E2B_GGUF=C:/.../gemma-4-E2B-it/gemma-4-E2B-it-Q8_0.gguf
//   set FLM_GEMMA4_E4B_GGUF=C:/.../gemma-4-E4B-it/gemma-4-E4B-it-Q8_0.gguf
//
// The files are opened READ-ONLY (GgufFile maps PAGE_READONLY) and nothing
// here writes, moves or deletes anything.

#include "models/gemma4/rai/aie_next/gemma4_rai_gguf.hpp"
#include "models/gemma4/rai/aie_next/gemma4_rai_host.hpp"
#include "rai/gguf_file.hpp"
#include "test_support.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
using flm::gemma4::ConfigFromMetadata;
using flm::rai::GgufFile;

std::filesystem::path g_e2b;
std::filesystem::path g_e4b;

/// \brief BF16, ggml type 30 -- the one non-F32, non-Q8_0 type Gemma 4 ships
constexpr std::uint32_t kGgmlBf16 = 30;

std::vector<std::int64_t> Shape(std::initializer_list<std::int64_t> dims) {
    return std::vector<std::int64_t>(dims);
}

// ---------------------------------------------------------------------------
// 1. The file opens at all.
//
// It did not, until this task. `per_layer_model_proj.weight` is BF16 on both
// rows -- the one BF16 tensor in either file -- and GgufFile knew only F32 and
// Q8_0, so `TensorByteLength` refused it and `Open` threw
// "per_layer_model_proj.weight: actual GGML type 30, expected F32 or Q8_0"
// before a single line of Gemma 4 code ran. That is why no test in this
// project had ever loaded a real Gemma 4 file, and it is why the transposed
// fixture survived as long as it did.

void TestRealE2bOpensAndItsOneBf16TensorReads() {
    const auto file = GgufFile::Open(g_e2b);
    TEST_REQUIRE(file->TensorCount() == 601);            // 6 + 35 * 17
    TEST_REQUIRE(file->String("general.architecture") == "gemma4");

    // The BF16 tensor, by shape and by accessor. 1536 x 8960 u16 elements;
    // RequireBf16 additionally proves the byte length GgufFile computed for
    // ggml type 30 is 2 bytes per element and not 4 or 34/32.
    const auto proj = file->RequireBf16("per_layer_model_proj.weight",
                                        Shape({8960, 1536}));
    TEST_REQUIRE(proj.values.size() == 8960u * 1536u);
    TEST_REQUIRE(proj.ggml_type == kGgmlBf16);

    // The other two types in the file, through their own accessors, so that
    // "BF16 was added" is distinguishable from "every type now reads as
    // anything".
    file->RequireQ8("token_embd.weight", Shape({262144, 1536}));
    file->RequireF32("output_norm.weight", Shape({1536}));
}

void TestRealE4bOpensAndItsOneBf16TensorReads() {
    const auto file = GgufFile::Open(g_e4b);
    TEST_REQUIRE(file->TensorCount() == 720);            // 6 + 42 * 17
    const auto proj = file->RequireBf16("per_layer_model_proj.weight",
                                        Shape({10752, 2560}));
    TEST_REQUIRE(proj.values.size() == 10752u * 2560u);
    TEST_REQUIRE(proj.ggml_type == kGgmlBf16);

    file->RequireQ8("token_embd.weight", Shape({262144, 2560}));
    file->RequireF32("output_norm.weight", Shape({2560}));
}

// Both directions of the type check, NAMING BOTH TYPES each way, the same as
// RequireQ8/RequireF32 do. The failure this rules out is the one that
// motivated the accessor: a byte length computed under the wrong type is not
// a crash, it is a view over the wrong bytes. The second half also proves
// `per_layer_model_proj.weight` really is BF16 in the file rather than
// something this reader quietly coerced.
void TestRequireBf16AndRequireF32RefuseEachOthersTensors() {
    const auto file = GgufFile::Open(g_e2b);
    auto error = RequireThrows([&] {
        file->RequireBf16("token_embd.weight", Shape({262144, 1536}));
    });
    RequireContains(error, "token_embd.weight");
    RequireContains(error, "Q8_0");
    RequireContains(error, "BF16");

    error = RequireThrows([&] {
        file->RequireF32("per_layer_model_proj.weight", Shape({8960, 1536}));
    });
    RequireContains(error, "per_layer_model_proj.weight");
    RequireContains(error, "BF16");
    RequireContains(error, "F32");
}

// ---------------------------------------------------------------------------
// 2. ORIENTATION: `ShapeOf` is [OUT, IN] on a real file.
//
// GGUF stores a tensor's dims fastest-varying first, so the FILE reads
// `blk.0.attn_q.weight = [1536, 2048]` -- input width first. `GgufFile`
// REVERSES that on the way in, so `ShapeOf` answers `[2048, 1536]`: output
// first. Phi-4, which has run against a real GGUF on hardware, agrees --
// phi4_rai.cpp requires `token_embd.weight` at `{kVocabularySize,
// kHiddenSize}`.
//
// The gemma4 fixture wrote the opposite, because Task C1 transcribed
// verified-gguf-facts.md's tables -- which quote FILE order -- straight into
// `AddTensor`, whose argument is in `ShapeOf` order. Three readers were then
// written against the fixture. Nothing could catch it, because nothing could
// open a real file.
//
// These assertions are the anchor. They are written from the dims the real
// files carry, REVERSED BY HAND, and they are what a future "simplification"
// back onto file order has to get past.

void RequireShape(const GgufFile& file, const std::string& name,
                  std::initializer_list<std::int64_t> expected) {
    const auto actual = file.Tensor(name).shape;
    const std::vector<std::int64_t> want(expected);
    if (actual != want) {
        std::string text = name + ": actual [";
        for (std::size_t i = 0; i < actual.size(); ++i)
            text += (i ? "," : "") + std::to_string(actual[i]);
        text += "], expected [";
        for (std::size_t i = 0; i < want.size(); ++i)
            text += (i ? "," : "") + std::to_string(want[i]);
        throw std::runtime_error(text + "]");
    }
}

void TestRealE2bShapeOfIsOutputWidthFirst() {
    const auto file = GgufFile::Open(g_e2b);
    // file dims [1536, 262144] -> [262144, 1536]
    RequireShape(*file, "token_embd.weight", {262144, 1536});
    // file dims [8960, 262144] -> [262144, 8960]
    RequireShape(*file, "per_layer_token_embd.weight", {262144, 8960});
    // file dims [1536, 8960] -> [8960, 1536]
    RequireShape(*file, "per_layer_model_proj.weight", {8960, 1536});

    // Attention, both geometries. Layer 0 slides, layer 4 is full-attention.
    RequireShape(*file, "blk.0.attn_q.weight", {2048, 1536});
    RequireShape(*file, "blk.0.attn_k.weight", {256, 1536});    // kv_heads 1
    RequireShape(*file, "blk.0.attn_v.weight", {256, 1536});
    RequireShape(*file, "blk.4.attn_q.weight", {4096, 1536});
    RequireShape(*file, "blk.4.attn_k.weight", {512, 1536});
    // THE ONLY PROJECTION WHOSE INPUT IS NOT `hidden`, and the one where
    // getting the convention backwards looks most plausible: attn_output
    // takes q_dim in and hidden out, so [out, in] is [1536, 2048].
    RequireShape(*file, "blk.0.attn_output.weight", {1536, 2048});
    RequireShape(*file, "blk.4.attn_output.weight", {1536, 4096});

    // MLP, both widths. The step is at layer 15 and is independent of the
    // attention period.
    RequireShape(*file, "blk.0.ffn_gate.weight", {6144, 1536});
    RequireShape(*file, "blk.0.ffn_up.weight", {6144, 1536});
    RequireShape(*file, "blk.0.ffn_down.weight", {1536, 6144});
    RequireShape(*file, "blk.15.ffn_gate.weight", {12288, 1536});
    RequireShape(*file, "blk.15.ffn_down.weight", {1536, 12288});

    // The PLE pair. These two are 1536x256 and 256x1536 -- the SAME element
    // count -- so a swap between them is not a size error anywhere, which is
    // why the engine demands the exact shapes.
    RequireShape(*file, "blk.0.inp_gate.weight", {256, 1536});
    RequireShape(*file, "blk.0.proj.weight", {1536, 256});

    // QK norms are sized to THEIR OWN LAYER'S head, not to a model constant.
    RequireShape(*file, "blk.0.attn_q_norm.weight", {256});
    RequireShape(*file, "blk.0.attn_k_norm.weight", {256});
    RequireShape(*file, "blk.4.attn_q_norm.weight", {512});
    RequireShape(*file, "blk.4.attn_k_norm.weight", {512});
}

void TestRealE4bShapeOfIsOutputWidthFirst() {
    const auto file = GgufFile::Open(g_e4b);
    RequireShape(*file, "token_embd.weight", {262144, 2560});
    RequireShape(*file, "per_layer_token_embd.weight", {262144, 10752});
    RequireShape(*file, "per_layer_model_proj.weight", {10752, 2560});

    // E4B has TWO KV heads, so its k/v widths are twice E2B's at the same
    // head size -- and layer 5, not layer 4, is the first full-attention one.
    RequireShape(*file, "blk.0.attn_q.weight", {2048, 2560});
    RequireShape(*file, "blk.0.attn_k.weight", {512, 2560});
    RequireShape(*file, "blk.5.attn_q.weight", {4096, 2560});
    RequireShape(*file, "blk.5.attn_k.weight", {1024, 2560});
    RequireShape(*file, "blk.0.attn_output.weight", {2560, 2048});

    RequireShape(*file, "blk.0.ffn_gate.weight", {10240, 2560});
    RequireShape(*file, "blk.41.ffn_gate.weight", {10240, 2560});   // uniform
    RequireShape(*file, "blk.0.ffn_down.weight", {2560, 10240});

    RequireShape(*file, "blk.0.inp_gate.weight", {256, 2560});
    RequireShape(*file, "blk.0.proj.weight", {2560, 256});
}

// ---------------------------------------------------------------------------
// 3. The whole derived config, against a HAND-WRITTEN table.
//
// Every number below is typed from verified-gguf-facts.md. NONE of it is
// computed from the file under test, and none of it is computed from the
// other model either -- the two rows disagree about the layer count, the
// hidden size, the KV head count, the FFN pattern, the full-attention period
// and the shared-cache pair, and a constant lifted from one is wrong about
// the other.

struct ExpectedConfig {
    std::int64_t layers, hidden, q_heads, kv_heads, head_dim, global_head_dim;
    std::int64_t kv_layers, ple_dim, sliding_window;
    std::vector<int> full_attention_layers;
    std::int64_t shared_sliding_owner, shared_full_owner;
};

// E2B: 35 layers, hidden 1536, one KV head, 15 owning caches (35 - 20
// sharing), FFN 6144 for layers 0-14 then 12288, full attention every 5th
// layer from 4, shared layers reading caches 13 (sliding) and 14 (full).
const ExpectedConfig kE2bExpected{
    35, 1536, 8, 1, 256, 512, 15, 256, 512, {4, 9, 14, 19, 24, 29, 34}, 13, 14};

// E4B: 42 layers, hidden 2560, TWO KV heads, 24 owning caches (42 - 18),
// a uniform 10240 FFN, full attention every 6th layer from 5, shared layers
// reading caches 22 and 23.
const ExpectedConfig kE4bExpected{
    42, 2560, 8, 2, 256, 512, 24, 256, 512, {5, 11, 17, 23, 29, 35, 41}, 22, 23};

void RequireConfigMatches(const flm::gemma4::Gemma4Config& cfg,
                          const ExpectedConfig& want, std::string_view row_name) {
    TEST_REQUIRE(cfg.row_name == row_name);
    TEST_REQUIRE(cfg.layers == want.layers);
    TEST_REQUIRE(cfg.hidden == want.hidden);
    TEST_REQUIRE(cfg.q_heads == want.q_heads);
    TEST_REQUIRE(cfg.kv_heads == want.kv_heads);
    TEST_REQUIRE(cfg.head_dim == want.head_dim);
    TEST_REQUIRE(cfg.global_head_dim == want.global_head_dim);
    // THE CORRECTION THAT MATTERS MOST: gemma4.attention.shared_kv_layers
    // counts the layers that SHARE (20 / 18), and kv_layers is the
    // complement. Taking the raw key would give 20 here and 18 on E4B, and
    // every shared layer would silently bind the wrong cache.
    TEST_REQUIRE(cfg.kv_layers == want.kv_layers);
    TEST_REQUIRE(cfg.ple_dim == want.ple_dim);
    TEST_REQUIRE(cfg.sliding_window == want.sliding_window);
    TEST_REQUIRE(cfg.vocab == 262144);
    TEST_REQUIRE(cfg.rope_theta == 1000000.0);        // full-attention layers
    TEST_REQUIRE(cfg.rope_theta_swa == 10000.0);      // sliding layers
    TEST_REQUIRE(cfg.logit_softcap == 30.0f);
    TEST_REQUIRE(cfg.eos_token_id == 106);
    TEST_REQUIRE(cfg.bos_token_id == 2);
    TEST_REQUIRE(cfg.add_bos);
    TEST_REQUIRE(cfg.group == 32 && cfg.head_group == 32 && cfg.ple_group == 32);
    // 1e-6, not Phi-4's 1e-5.
    TEST_REQUIRE(cfg.eps > 9.0e-7f && cfg.eps < 1.1e-6f);

    TEST_REQUIRE(static_cast<std::int64_t>(cfg.layer_is_swa.size()) == want.layers);
    TEST_REQUIRE(static_cast<std::int64_t>(cfg.layer_intermediate.size()) == want.layers);
    TEST_REQUIRE(static_cast<std::int64_t>(cfg.layer_cache_owner.size()) == want.layers);
    for (std::int64_t layer = 0; layer < want.layers; ++layer) {
        const auto index = static_cast<std::size_t>(layer);
        const bool is_full =
            std::find(want.full_attention_layers.begin(),
                      want.full_attention_layers.end(),
                      static_cast<int>(layer)) != want.full_attention_layers.end();
        TEST_REQUIRE(cfg.layer_is_swa[index] == !is_full);
        TEST_REQUIRE(cfg.LayerHeadDim(layer) == (is_full ? 512 : 256));
        if (layer < want.kv_layers) {
            TEST_REQUIRE(cfg.layer_cache_owner[index] == layer);
        } else {
            TEST_REQUIRE(cfg.layer_cache_owner[index] ==
                         (is_full ? want.shared_full_owner : want.shared_sliding_owner));
        }
    }
}

void TestRealE2bConfigMatchesTheHandWrittenTable() {
    const auto file = GgufFile::Open(g_e2b);
    const auto cfg = ConfigFromMetadata(*file);
    RequireConfigMatches(cfg, kE2bExpected, "E2B");
    // E2B's FFN width STEPS, and it steps at 15 -- which
    // gemma4.feed_forward_length's own 35-entry array also says, but
    // config.json's `intermediate_size: 6144` does not.
    for (std::size_t layer = 0; layer < 35; ++layer) {
        TEST_REQUIRE(cfg.layer_intermediate[layer] == (layer < 15 ? 6144 : 12288));
    }
}

void TestRealE4bConfigMatchesTheHandWrittenTable() {
    const auto file = GgufFile::Open(g_e4b);
    const auto cfg = ConfigFromMetadata(*file);
    RequireConfigMatches(cfg, kE4bExpected, "E4B");
    for (std::size_t layer = 0; layer < 42; ++layer) {
        TEST_REQUIRE(cfg.layer_intermediate[layer] == 10240);
    }
}

// ---------------------------------------------------------------------------
// 4. The ENGINE's expected shapes, on the real files.
//
// `gemma4_rai.cpp` demands an exact shape for every weight it packs, which
// is good -- a wrong convention throws naming the tensor and both shapes
// rather than packing something plausible. But those demands were checked
// only against the fixture, and the fixture was transposed, so "the engine
// agrees with the fixture" proved nothing at all.
//
// The engine cannot be linked here (it needs corelib, boost and XRT), so
// this reproduces its per-weight expectations as a table and applies them to
// EVERY LAYER of both real files. It is a duplicate of a rule rather than a
// call into it -- which is a real limitation, stated rather than hidden: if
// someone changes gemma4_rai.cpp's shapes and not these, this does not
// notice. What it does catch, and what nothing else in the suite can, is the
// two of them agreeing with each other and disagreeing with Google.

void RequireEngineShapesHold(const GgufFile& file,
                             const flm::gemma4::Gemma4Config& cfg) {
    // Model-level: lm_head is TIED to token_embd, so this one tensor is both
    // the embedding table and the widest matmul in the model.
    file.RequireQ8("token_embd.weight", Shape({cfg.vocab, cfg.hidden}));
    file.RequireF32("output_norm.weight", Shape({cfg.hidden}));

    for (std::int64_t layer = 0; layer < cfg.layers; ++layer) {
        const auto prefix = "blk." + std::to_string(layer) + ".";
        const auto head_dim = cfg.LayerHeadDim(layer);
        const auto q_dim = cfg.q_heads * head_dim;
        const auto kv_dim = cfg.kv_heads * head_dim;
        const auto intermediate = cfg.layer_intermediate[static_cast<std::size_t>(layer)];

        file.RequireQ8(prefix + "attn_q.weight", Shape({q_dim, cfg.hidden}));
        file.RequireQ8(prefix + "attn_k.weight", Shape({kv_dim, cfg.hidden}));
        file.RequireQ8(prefix + "attn_v.weight", Shape({kv_dim, cfg.hidden}));
        file.RequireQ8(prefix + "attn_output.weight", Shape({cfg.hidden, q_dim}));
        file.RequireQ8(prefix + "ffn_gate.weight", Shape({intermediate, cfg.hidden}));
        file.RequireQ8(prefix + "ffn_up.weight", Shape({intermediate, cfg.hidden}));
        file.RequireQ8(prefix + "ffn_down.weight", Shape({cfg.hidden, intermediate}));

        // The QK norms are sized to THIS LAYER's head, not the model's.
        file.RequireF32(prefix + "attn_q_norm.weight", Shape({head_dim}));
        file.RequireF32(prefix + "attn_k_norm.weight", Shape({head_dim}));
        // The five per-layer norms. `post_norm` is the fifth -- NOT a
        // `per_layer_norm`, which was Task C1's guess and does not exist.
        for (const char* norm : {"attn_norm.weight", "post_attention_norm.weight",
                                 "ffn_norm.weight", "post_ffw_norm.weight",
                                 "post_norm.weight"}) {
            file.RequireF32(prefix + norm, Shape({cfg.hidden}));
        }
        // The PLE trio. inp_gate and proj are F32, NOT Q8_0 -- corelib's ple
        // packer takes `const float*` and requantizes itself, so there is no
        // Q8_0 route for either.
        file.RequireF32(prefix + "inp_gate.weight", Shape({cfg.ple_dim, cfg.hidden}));
        file.RequireF32(prefix + "proj.weight", Shape({cfg.hidden, cfg.ple_dim}));
        file.RequireF32(prefix + "layer_output_scale.weight", Shape({1}));
    }

    // The last layer's ple takes `output_norm.weight` as its next_norm,
    // because there is no blk.{layers}.attn_norm. Assert the absence, since
    // an engine that looked for one would fail only on the final layer.
    TEST_REQUIRE(!file.HasTensor("blk." + std::to_string(cfg.layers) + ".attn_norm.weight"));
    // lm_head is tied on both rows.
    TEST_REQUIRE(!file.HasTensor("output.weight"));
}

void TestEngineWeightShapesHoldOnTheRealE2b() {
    const auto file = GgufFile::Open(g_e2b);
    RequireEngineShapesHold(*file, ConfigFromMetadata(*file));
}

void TestEngineWeightShapesHoldOnTheRealE4b() {
    const auto file = GgufFile::Open(g_e4b);
    RequireEngineShapesHold(*file, ConfigFromMetadata(*file));
}

// ---------------------------------------------------------------------------
// 5. `GatherPerLayerEmbedding`'s MEMORY LAYOUT, against the reference driver.
//
// This is a DIFFERENT question from the `ShapeOf` convention above, and
// reversing dims does not answer it. `per_layer_token_embd.weight`'s file
// dims are [layers * ple_dim, vocab], and dims[0] is the FASTEST-VARYING
// extent -- so the bytes are `vocab` rows of `layers * ple_dim` contiguous
// elements, with the Q8_0 blocks running along the PLE axis. One token's
// whole per-layer slice is ONE CONTIGUOUS ROW of 280 blocks (E2B) or 336
// (E4B).
//
// Task 15 read it as `layers * ple_dim` rows of `vocab` instead -- one block
// picked out of each of 8960 rows -- which verified-gguf-facts.md endorsed in
// so many words ("No change is needed to GatherPerLayerEmbedding"). THE TWO
// LAYOUTS HAVE THE IDENTICAL TOTAL BYTE COUNT, so every length check passes
// and the function returns 8960 finite, plausible floats that are simply the
// wrong ones.
//
// The expected values below are dumped from the REFERENCE DRIVER's own rule
// -- gemma4_driver.py's `per_layer_embedding()` builds
// `LazyEmbedding(data, dtype, (dims[1], dims[0]))` and slices row i at
// `i * (nbytes // vocab)` -- applied to these exact files. They are
// hardcoded here, not recomputed, so this test cannot agree with a wrong
// implementation by sharing its assumption.

void RequireClose(float actual, double expected, const char* what) {
    // Q8_0 codes times an fp16 scale: exact to well within this.
    if (!(std::abs(static_cast<double>(actual) - expected) < 1e-6)) {
        throw std::runtime_error(std::string(what) + ": actual " +
                                 std::to_string(actual) + ", expected " +
                                 std::to_string(expected));
    }
}

void RequireGatherMatchesTheDriver(const GgufFile& file, std::int64_t layers,
                                   std::int64_t hidden_unused,
                                   const std::vector<double>& first_four,
                                   const std::vector<double>& last_four) {
    (void)hidden_unused;
    constexpr std::int64_t kPleDim = 256;
    constexpr std::int64_t kVocab = 262144;
    constexpr std::int64_t kBosToken = 2;
    const auto width = layers * kPleDim;
    const auto table = file.RequireQ8("per_layer_token_embd.weight",
                                      Shape({kVocab, width}));
    std::vector<float> out(static_cast<std::size_t>(width));
    flm::gemma4::GatherPerLayerEmbedding(table.bytes, layers, kPleDim, kVocab,
                                         kBosToken, out);
    for (std::size_t i = 0; i < 4; ++i) {
        RequireClose(out[i], first_four[i], "per-layer embedding head");
        RequireClose(out[out.size() - 4 + i], last_four[i],
                     "per-layer embedding tail");
    }
    // Not all one value, and not all zero -- a layout error that happened to
    // land on a constant region would otherwise slip past four samples.
    TEST_REQUIRE(std::any_of(out.begin(), out.end(),
                             [&](float v) { return v != out[0]; }));
}

void TestGatherPerLayerEmbeddingMatchesTheDriverOnRealE2b() {
    const auto file = GgufFile::Open(g_e2b);
    // Token 2 (<bos>) of gemma-4-E2B-it-Q8_0.gguf, 8960 wide.
    RequireGatherMatchesTheDriver(
        *file, 35, 1536,
        {0.060577393, -0.033317566, -0.0090866089, -0.012115479},
        {-0.15021229, -0.021652222, 0.050070763, 0.040597916});
}

/// \brief the WHOLE per-layer-input formula, on a real file, against a
///        reference computed outside this codebase
///
/// THE ONLY PLACE IN THIS SUITE WHERE A FORWARD PASS'S ARITHMETIC IS COMPARED
/// TO ANYTHING. Everything else the engine does is measured against a FAKE
/// corelib that moves no bytes: the dispatch sequence, the operand wiring, the
/// descriptors and the caches are all checked, and not one number is. The
/// per-layer inputs are the exception, because they are computed on the HOST
/// -- `per_layer_model_proj` is unreachable at this model's PDI pair and
/// llama.cpp puts the same matmul on the CPU regardless -- so they can be
/// checked here and nowhere else.
///
/// The expected values were produced by an independent NumPy implementation of
/// the reference driver's `_ple_inputs`, reading the real file's own bytes:
/// decode token 2's `token_embd` row and scale by sqrt(hidden); project
/// through `per_layer_model_proj` (BF16, read as `layers*ple_dim` rows of
/// `hidden`) and scale by 1/sqrt(hidden); RMS-normalize each layer's slice
/// with `per_layer_proj_norm` at the file's own epsilon; add the gathered
/// per-layer embedding scaled by sqrt(ple_dim); halve by 1/sqrt(2).
///
/// WHAT THIS CATCHES THAT NOTHING ELSE CAN: a transposed read of
/// `per_layer_model_proj` (same element count, finite, plausible), a missing
/// or doubled scale factor (a wrong magnitude, fluent text), the epsilon read
/// from the wrong place, and a per-layer slice normalized over the wrong
/// extent. All four are silent against the synthetic fixture, whose Q8_0
/// payloads decode to zeros and whose BF16 tensor is zero-filled.
void RequirePerLayerInputsMatchTheReference(
    const GgufFile& file, std::int64_t layers, std::int64_t hidden,
    const double (&layer0_first_four)[4], const double (&layer0_last_four)[4],
    const double (&layer_last_first_four)[4],
    const double (&layer_last_last_four)[4]) {
    constexpr std::int64_t kPleDim = 256;
    constexpr std::int64_t kVocab = 262144;
    constexpr int kBosToken = 2;
    const auto width = layers * kPleDim;

    const auto embedding = file.RequireQ8("token_embd.weight", Shape({kVocab, hidden}));
    const auto table =
        file.RequireQ8("per_layer_token_embd.weight", Shape({kVocab, width}));
    const auto projection =
        file.RequireBf16("per_layer_model_proj.weight", Shape({width, hidden}));
    const auto gamma =
        file.RequireF32("per_layer_proj_norm.weight", Shape({kPleDim}));
    const auto epsilon = static_cast<float>(
        file.Number("gemma4.attention.layer_norm_rms_epsilon"));

    // The scaled embedding row, exactly as the engine builds it: one
    // contiguous Q8_0 row, decoded, times sqrt(hidden).
    const auto row_bytes = static_cast<std::size_t>(hidden / 32) * 34;
    std::vector<float> scaled(static_cast<std::size_t>(hidden));
    flm::gemma4::DecodeQ8Row(
        embedding.bytes.subspan(static_cast<std::size_t>(kBosToken) * row_bytes,
                                row_bytes),
        scaled);
    const auto embedding_scale =
        static_cast<float>(std::sqrt(static_cast<double>(hidden)));
    for (float& value : scaled) value *= embedding_scale;

    const int ids[1] = {kBosToken};
    const auto planes = flm::gemma4::BuildPerLayerInputs(
        table.bytes, projection.values, gamma.values, scaled, ids, layers,
        kPleDim, hidden, kVocab, epsilon);

    TEST_REQUIRE(planes.size() == static_cast<std::size_t>(layers));
    const auto check = [&](std::size_t layer, const double (&first)[4],
                           const double (&last)[4]) {
        const auto& plane = planes[layer];
        TEST_REQUIRE(plane.size() == static_cast<std::size_t>(kPleDim));
        for (std::size_t i = 0; i < 4; ++i) {
            RequireClose(plane[i], first[i], "per-layer input head");
            RequireClose(plane[plane.size() - 4 + i], last[i],
                         "per-layer input tail");
        }
    };
    // LAYER 0 AND THE LAST LAYER, because a reader that projected the whole
    // width in one go, or reused one slice for every layer, agrees on one of
    // them and not on both.
    check(0, layer0_first_four, layer0_last_four);
    check(static_cast<std::size_t>(layers - 1), layer_last_first_four,
          layer_last_last_four);
    TEST_REQUIRE(planes[0] != planes[static_cast<std::size_t>(layers - 1)]);
}

void TestPerLayerInputsMatchTheReferenceOnRealE2b() {
    const auto file = GgufFile::Open(g_e2b);
    RequirePerLayerInputsMatchTheReference(
        *file, 35, 1536,
        {0.559545735, -0.412504045, -0.115115489, 0.0566222885},
        {-0.229557123, 0.0139010772, 0.353591797, 0.103018071},
        {0.312784019, -0.518369327, -0.125572238, 0.0633239999},
        {0.45440066, -0.371125478, 0.405698639, 0.447286685});
}

void TestPerLayerInputsMatchTheReferenceOnRealE4b() {
    const auto file = GgufFile::Open(g_e4b);
    RequirePerLayerInputsMatchTheReference(
        *file, 42, 2560,
        {-0.213408942, 0.413220405, 0.0715314188, -0.404233209},
        {0.0789781903, 0.0489501076, -0.515433868, 0.147717535},
        {1.37376722, 0.202201064, 0.121642122, 2.93530821},
        {-0.130365683, -0.0349114712, -0.156501543, -0.367043872});
}

void TestGatherPerLayerEmbeddingMatchesTheDriverOnRealE4b() {
    const auto file = GgufFile::Open(g_e4b);
    // Token 2 (<bos>) of gemma-4-E4B-it-Q8_0.gguf, 10752 wide.
    RequireGatherMatchesTheDriver(
        *file, 42, 2560,
        {-0.016731262, 0.040351868, -0.0068893433, -0.035430908},
        {-0.016489029, 0.012683868, -0.035514832, -0.022830963});
}

}  // namespace

int main() {
    const char* e2b = std::getenv("FLM_GEMMA4_E2B_GGUF");
    const char* e4b = std::getenv("FLM_GEMMA4_E4B_GGUF");
    const bool have_e2b = e2b != nullptr && *e2b != '\0';
    const bool have_e4b = e4b != nullptr && *e4b != '\0';

    if (!have_e2b && !have_e4b) {
        std::cout
            << "SKIP: neither FLM_GEMMA4_E2B_GGUF nor FLM_GEMMA4_E4B_GGUF is "
               "set, so NO TEST IN THIS PROJECT HAS READ A REAL GEMMA 4 FILE "
               "IN THIS RUN. Every other gemma4 target ran against the "
               "synthetic fixture only. Set both to the real "
               "gemma-4-E{2,4}B-it-Q8_0.gguf paths to cover the orientation "
               "and BF16 assertions this binary owns.\n";
        return 77;
    }
    // Deliberately a FAILURE, not a partial run: half these assertions
    // passing is indistinguishable from all of them passing in a ctest line,
    // and the two models are here precisely because each one catches what the
    // other cannot (E2B's stepping FFN, E4B's different full-attention
    // period and its second KV head).
    if (!have_e2b || !have_e4b) {
        std::cerr << "FAIL: exactly one of FLM_GEMMA4_E2B_GGUF / "
                     "FLM_GEMMA4_E4B_GGUF is set. Set both or neither -- a "
                     "half-covered run that reports green is what this binary "
                     "exists to prevent.\n";
        return 1;
    }

    g_e2b = std::filesystem::path(e2b);
    g_e4b = std::filesystem::path(e4b);
    for (const auto& path : {g_e2b, g_e4b}) {
        if (!std::filesystem::exists(path)) {
            std::cerr << "FAIL: " << path.string() << " does not exist.\n";
            return 1;
        }
    }
    std::cout << "REAL FILES: " << g_e2b.string() << "\n             "
              << g_e4b.string() << '\n';

#define RUN_TEST(name) RunTest(&name, #name)
    RUN_TEST(TestRealE2bOpensAndItsOneBf16TensorReads);
    RUN_TEST(TestRealE4bOpensAndItsOneBf16TensorReads);
    RUN_TEST(TestRequireBf16AndRequireF32RefuseEachOthersTensors);
    RUN_TEST(TestRealE2bShapeOfIsOutputWidthFirst);
    RUN_TEST(TestRealE4bShapeOfIsOutputWidthFirst);
    RUN_TEST(TestRealE2bConfigMatchesTheHandWrittenTable);
    RUN_TEST(TestRealE4bConfigMatchesTheHandWrittenTable);
    RUN_TEST(TestEngineWeightShapesHoldOnTheRealE2b);
    RUN_TEST(TestEngineWeightShapesHoldOnTheRealE4b);
    RUN_TEST(TestGatherPerLayerEmbeddingMatchesTheDriverOnRealE2b);
    RUN_TEST(TestGatherPerLayerEmbeddingMatchesTheDriverOnRealE4b);
    RUN_TEST(TestPerLayerInputsMatchTheReferenceOnRealE2b);
    RUN_TEST(TestPerLayerInputsMatchTheReferenceOnRealE4b);
#undef RUN_TEST
}
