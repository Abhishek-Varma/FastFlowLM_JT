#include "models/gemma4/rai/aie_next/gemma4_rai_gguf.hpp"
#include "gemma4_gguf_fixture.hpp"
#include "rai/gguf_file.hpp"
#include "test_support.hpp"

#include <chrono>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <vector>

namespace {
using flm::gemma4::RequireRowFor;

void TestBothShippedRowsAreFound() {
    const auto& e2b = RequireRowFor(35, 1536, 8, 1, 256, 512);
    TEST_REQUIRE(e2b.name == "E2B");
    TEST_REQUIRE(e2b.vocab == 262144);
    TEST_REQUIRE(e2b.group == 32);
    TEST_REQUIRE(e2b.head_group == 32);
    TEST_REQUIRE(e2b.ple_group == 32);

    const auto& e4b = RequireRowFor(42, 2560, 8, 2, 256, 512);
    TEST_REQUIRE(e4b.name == "E4B");
    TEST_REQUIRE(e4b.vocab == 262144);
    TEST_REQUIRE(e4b.group == 32);
}

void TestAnUnknownShapeAsksForARowRatherThanGuessing() {
    // Gemma 4 12B: 48 layers, hidden 3840, 16 q heads, and it DROPS to 1 KV
    // head on its full-attention layers. Its groups have never been read off an
    // ELF set, so guessing a row would replace this clear rejection with a
    // dispatch failure a long way from here.
    const auto error = RequireThrows([&] {
        RequireRowFor(48, 3840, 16, 8, 256, 512);
    });
    RequireContains(error, "48");
    RequireContains(error, "3840");
    RequireContains(error, "add a row");
}

// The row table above is only half of Task C1: the fixture that will feed
// every later Gemma 4 task (FFN-width derivation, SWA-layer derivation,
// cache ownership, the whole-package contract) has to actually produce the
// files verified-gguf-facts.md measured -- 601 tensors on E2B, 720 on E4B,
// and gemma4.attention.shared_kv_layers written with the FILE's sharing
// semantics (20/18), not the owning counts (15/24) the old plan confused it
// with. These tests assert that directly, rather than leaving "assert this
// yourself while writing it" as something only a comment claims happened.

void TestE2bFixtureMatchesTheRealFilesShapeAndTensorCount() {
    const auto options = gemma4_fixture::E2bOptions();
    auto builder = gemma4_fixture::MakeBuilder(options);
    // 6 file-level tensors + 17 per layer, asserted before it ever touches
    // disk -- see verified-gguf-facts.md, "tensor count 6 + layers*17".
    TEST_REQUIRE(builder.TensorCount() == 6 + 35 * 17);
    TEST_REQUIRE(builder.TensorCount() == 601);

    const auto path = gemma4_fixture::Write(options);
    const auto file = flm::rai::GgufFile::Open(path);
    TEST_REQUIRE(file->TensorCount() == 601);
    TEST_REQUIRE(file->String("general.architecture") == "gemma4");
    TEST_REQUIRE(file->Unsigned("gemma4.block_count") == 35);
    TEST_REQUIRE(file->Unsigned("gemma4.embedding_length") == 1536);
    TEST_REQUIRE(file->Unsigned("gemma4.attention.head_count_kv") == 1);
    TEST_REQUIRE(file->Unsigned("gemma4.attention.key_length") == 512);
    TEST_REQUIRE(file->Unsigned("gemma4.attention.key_length_swa") == 256);
    // The correction that matters most: 20, the SHARING count, never 15 --
    // the OWNING count (35 - 20 = 15) that the superseded plan asked for.
    TEST_REQUIRE(file->Unsigned("gemma4.attention.shared_kv_layers") == 20);
    TEST_REQUIRE(file->Number("gemma4.final_logit_softcapping") == 30.0);
    TEST_REQUIRE(file->Unsigned("tokenizer.ggml.eos_token_id") == 106);
    TEST_REQUIRE(file->Unsigned("tokenizer.ggml.bos_token_id") == 2);
    TEST_REQUIRE(file->Boolean("tokenizer.ggml.add_bos_token"));
    TEST_REQUIRE(!file->HasTensor("output.weight"));

    // E2B is the ARRAY case: 35 entries, stepping from 6144 to 12288 at
    // layer 15 -- exactly what the real E2B GGUF carries.
    const auto ffn_lengths = file->UnsignedArray("gemma4.feed_forward_length");
    TEST_REQUIRE(ffn_lengths.size() == 35);
    TEST_REQUIRE(ffn_lengths[0] == 6144);
    TEST_REQUIRE(ffn_lengths[14] == 6144);
    TEST_REQUIRE(ffn_lengths[15] == 12288);
    TEST_REQUIRE(ffn_lengths[34] == 12288);

    // ShapeOf[0] is the OUTPUT width throughout, because `ShapeOf` reverses
    // the GGUF's own dims and the file stores them input-first. attn_q steps
    // 2048 (sliding) / 4096 (full) at hidden 1536, and ffn_gate's own shape
    // is what a real reader derives width from -- never the metadata array
    // above. ShapeOf[1] is `hidden` on every one of these, which is what a
    // reader that had the convention backwards used to read.
    TEST_REQUIRE(file->Tensor("blk.4.attn_q.weight").shape[0] == 4096);   // full
    TEST_REQUIRE(file->Tensor("blk.4.attn_q.weight").shape[1] == 1536);   // in
    TEST_REQUIRE(file->Tensor("blk.5.attn_q.weight").shape[0] == 2048);   // sliding
    TEST_REQUIRE(file->Tensor("blk.14.ffn_gate.weight").shape[0] == 6144);
    TEST_REQUIRE(file->Tensor("blk.15.ffn_gate.weight").shape[0] == 12288);
    // per_layer_token_embd folds the layer axis into the file's fastest-
    // varying extent, so after reversal it is ShapeOf[1], not ShapeOf[0].
    TEST_REQUIRE(file->Tensor("per_layer_token_embd.weight").shape[0] == 262144);
    TEST_REQUIRE(file->Tensor("per_layer_token_embd.weight").shape[1] == 35 * 256);
}

void TestE4bFixtureMatchesTheRealFilesShapeAndTensorCount() {
    const auto options = gemma4_fixture::E4bOptions();
    auto builder = gemma4_fixture::MakeBuilder(options);
    TEST_REQUIRE(builder.TensorCount() == 6 + 42 * 17);
    TEST_REQUIRE(builder.TensorCount() == 720);

    const auto path = gemma4_fixture::Write(options);
    const auto file = flm::rai::GgufFile::Open(path);
    TEST_REQUIRE(file->TensorCount() == 720);
    TEST_REQUIRE(file->Unsigned("gemma4.block_count") == 42);
    TEST_REQUIRE(file->Unsigned("gemma4.embedding_length") == 2560);
    TEST_REQUIRE(file->Unsigned("gemma4.attention.head_count_kv") == 2);
    // 18, the SHARING count -- not 24, the OWNING count (42 - 18 = 24).
    TEST_REQUIRE(file->Unsigned("gemma4.attention.shared_kv_layers") == 18);
    TEST_REQUIRE(!file->HasTensor("output.weight"));

    // E4B is the SCALAR case: one value, 10240, for every layer.
    TEST_REQUIRE(file->Unsigned("gemma4.feed_forward_length") == 10240);
    TEST_REQUIRE(file->Tensor("blk.0.ffn_gate.weight").shape[0] == 10240);
    TEST_REQUIRE(file->Tensor("blk.0.ffn_gate.weight").shape[1] == 2560);
    TEST_REQUIRE(file->Tensor("blk.41.ffn_gate.weight").shape[0] == 10240);
    TEST_REQUIRE(file->Tensor("per_layer_token_embd.weight").shape[1] == 42 * 256);
}

// The fixture must let a later task prove its reader ignores
// gemma4.feed_forward_length entirely, by setting the metadata to something
// that disagrees with the tensor shapes MakeBuilder already committed --
// without touching a single tensor. MakeBuilder returning the Builder before
// Write() is what makes that possible.
void TestFeedForwardLengthMetadataIsSettableIndependentlyOfTensorShapes() {
    auto options = gemma4_fixture::E4bOptions();
    auto builder = gemma4_fixture::MakeBuilder(options);
    const auto before = builder.TensorCount();
    builder.SetMetadata("gemma4.feed_forward_length", std::uint32_t{99999});
    TEST_REQUIRE(builder.TensorCount() == before);   // no tensor touched

    const auto path = builder.Write();
    const auto file = flm::rai::GgufFile::Open(path);
    TEST_REQUIRE(file->Unsigned("gemma4.feed_forward_length") == 99999);
    // The tensor shape a real reader would actually use is untouched.
    TEST_REQUIRE(file->Tensor("blk.0.ffn_gate.weight").shape[0] == 10240);
}

// ---------------------------------------------------------------------------
// The complete table, pinned.
//
// Everything above checks 2 of 17 per-layer tensors and 1 of 6 file-level
// ones. The other 15 per-layer tensors -- including post_norm and
// layer_output_scale, the two fields that were WRONG until a hand dump of the
// real file caught them -- had zero coverage: nothing here would have caught
// either mistake, and TensorCount() alone stays 601/720 through any rename or
// reshape, because it counts tensors, not names.
//
// EVERY SHAPE BELOW IS IN `GgufFile::ShapeOf`'s CONVENTION, WHICH IS
// [OUT, IN] -- the REVERSE of the dims a GGUF stores and the reverse of the
// tables in verified-gguf-facts.md, which quote file order. That distinction
// is what this whole file got wrong until Task R1: the tables were
// transcribed from that document verbatim into a fixture whose `AddTensor`
// takes ShapeOf order, so the fixture, these tables and Tasks C2-C5's
// readers were all consistently transposed and confirmed each other.
// `blk.0.attn_q.weight` reads [1536, 2048] in the file and is written
// {2048, 1536} below. test_gemma4_real_gguf.cpp asserts the same numbers
// against the real gemma-4-E{2,4}B-it-Q8_0.gguf, which is the only thing
// that can catch this class of error at all.
//
// Each table below is written down independently of gemma4_gguf_fixture.hpp:
// literal numbers, not a call into MakeBuilder/Write, and not the same
// formula re-typed -- so a bug shared between the fixture's shape
// computation and this table's numbers cannot cancel out. Provenance:
//   - E2B layer 0 (sliding, low FFN) is the exact dump in
//     verified-gguf-facts.md's "Layer 0, all 17 tensors" section --
//     every number here is copied from that dump, not computed.
//   - E2B's attn_q full-attention width (4096) and the FFN step to 12288 at
//     layer 15 are each independently confirmed in verified-gguf-facts.md's
//     "other shapes" section. E4B's attn_q (2048/4096) and uniform ffn_gate
//     (10240) are confirmed there too.
//   - The remaining numbers (attn_k/attn_v/attn_output widths at a
//     full-attention layer, attn_q_norm/attn_k_norm at head_dim 512, and
//     every E4B norm/PLE shape beyond what's listed above) were derived by
//     Task C1 from structural rules rather than measured, because no real
//     GGUF could be opened then. TASK R1 RE-DUMPED BOTH REAL FILES and
//     every one of them holds: kv width is kv_heads * head_dim (so E4B's
//     attn_k is 512/1024 where E2B's is 256/512), a QK norm is sized to its
//     own layer's head_dim, attn_output and ffn_down really are the two
//     whose input is not `hidden`, and the PLE trio and the five generic
//     norms do not vary with attention kind. What the re-dump did NOT
//     confirm is the ORIENTATION these were all written in; see the note
//     above.
struct ExpectedTensor {
    std::string_view name;   // without the "blk.N." prefix, for per-layer rows
    std::vector<std::uint64_t> shape;
    std::uint32_t type;
};

using gemma4_fixture::kBf16;
using gemma4_fixture::kF32;
using gemma4_fixture::kQ8_0;

// E2B layer 0: sliding (head_dim 256), low FFN (6144) -- the literal dump.
const std::vector<ExpectedTensor> kE2bLayer0Sliding = {
    {"attn_q.weight", {2048, 1536}, kQ8_0},
    {"attn_k.weight", {256, 1536}, kQ8_0},
    {"attn_v.weight", {256, 1536}, kQ8_0},
    {"attn_output.weight", {1536, 2048}, kQ8_0},
    {"ffn_gate.weight", {6144, 1536}, kQ8_0},
    {"ffn_up.weight", {6144, 1536}, kQ8_0},
    {"ffn_down.weight", {1536, 6144}, kQ8_0},
    {"attn_norm.weight", {1536}, kF32},
    {"post_attention_norm.weight", {1536}, kF32},
    {"ffn_norm.weight", {1536}, kF32},
    {"post_ffw_norm.weight", {1536}, kF32},
    {"post_norm.weight", {1536}, kF32},
    {"attn_q_norm.weight", {256}, kF32},
    {"attn_k_norm.weight", {256}, kF32},
    {"inp_gate.weight", {256, 1536}, kF32},
    {"proj.weight", {1536, 256}, kF32},
    {"layer_output_scale.weight", {1}, kF32},
};

// E2B layer 4: full-attention (head_dim 512), low FFN (6144, layer < 15).
const std::vector<ExpectedTensor> kE2bLayer4Full = {
    {"attn_q.weight", {4096, 1536}, kQ8_0},
    {"attn_k.weight", {512, 1536}, kQ8_0},
    {"attn_v.weight", {512, 1536}, kQ8_0},
    {"attn_output.weight", {1536, 4096}, kQ8_0},
    {"ffn_gate.weight", {6144, 1536}, kQ8_0},
    {"ffn_up.weight", {6144, 1536}, kQ8_0},
    {"ffn_down.weight", {1536, 6144}, kQ8_0},
    {"attn_norm.weight", {1536}, kF32},
    {"post_attention_norm.weight", {1536}, kF32},
    {"ffn_norm.weight", {1536}, kF32},
    {"post_ffw_norm.weight", {1536}, kF32},
    {"post_norm.weight", {1536}, kF32},
    {"attn_q_norm.weight", {512}, kF32},
    {"attn_k_norm.weight", {512}, kF32},
    {"inp_gate.weight", {256, 1536}, kF32},
    {"proj.weight", {1536, 256}, kF32},
    {"layer_output_scale.weight", {1}, kF32},
};

// E2B layer 15: sliding (head_dim 256), high FFN (12288, layer >= 15) --
// proves the FFN step and the attention period are independent axes.
const std::vector<ExpectedTensor> kE2bLayer15Sliding = {
    {"attn_q.weight", {2048, 1536}, kQ8_0},
    {"attn_k.weight", {256, 1536}, kQ8_0},
    {"attn_v.weight", {256, 1536}, kQ8_0},
    {"attn_output.weight", {1536, 2048}, kQ8_0},
    {"ffn_gate.weight", {12288, 1536}, kQ8_0},
    {"ffn_up.weight", {12288, 1536}, kQ8_0},
    {"ffn_down.weight", {1536, 12288}, kQ8_0},
    {"attn_norm.weight", {1536}, kF32},
    {"post_attention_norm.weight", {1536}, kF32},
    {"ffn_norm.weight", {1536}, kF32},
    {"post_ffw_norm.weight", {1536}, kF32},
    {"post_norm.weight", {1536}, kF32},
    {"attn_q_norm.weight", {256}, kF32},
    {"attn_k_norm.weight", {256}, kF32},
    {"inp_gate.weight", {256, 1536}, kF32},
    {"proj.weight", {1536, 256}, kF32},
    {"layer_output_scale.weight", {1}, kF32},
};

// E2B layer 19: full-attention, high FFN.
const std::vector<ExpectedTensor> kE2bLayer19Full = {
    {"attn_q.weight", {4096, 1536}, kQ8_0},
    {"attn_k.weight", {512, 1536}, kQ8_0},
    {"attn_v.weight", {512, 1536}, kQ8_0},
    {"attn_output.weight", {1536, 4096}, kQ8_0},
    {"ffn_gate.weight", {12288, 1536}, kQ8_0},
    {"ffn_up.weight", {12288, 1536}, kQ8_0},
    {"ffn_down.weight", {1536, 12288}, kQ8_0},
    {"attn_norm.weight", {1536}, kF32},
    {"post_attention_norm.weight", {1536}, kF32},
    {"ffn_norm.weight", {1536}, kF32},
    {"post_ffw_norm.weight", {1536}, kF32},
    {"post_norm.weight", {1536}, kF32},
    {"attn_q_norm.weight", {512}, kF32},
    {"attn_k_norm.weight", {512}, kF32},
    {"inp_gate.weight", {256, 1536}, kF32},
    {"proj.weight", {1536, 256}, kF32},
    {"layer_output_scale.weight", {1}, kF32},
};

// E4B layer 0: sliding (head_dim 256).
const std::vector<ExpectedTensor> kE4bLayer0Sliding = {
    {"attn_q.weight", {2048, 2560}, kQ8_0},
    {"attn_k.weight", {512, 2560}, kQ8_0},
    {"attn_v.weight", {512, 2560}, kQ8_0},
    {"attn_output.weight", {2560, 2048}, kQ8_0},
    {"ffn_gate.weight", {10240, 2560}, kQ8_0},
    {"ffn_up.weight", {10240, 2560}, kQ8_0},
    {"ffn_down.weight", {2560, 10240}, kQ8_0},
    {"attn_norm.weight", {2560}, kF32},
    {"post_attention_norm.weight", {2560}, kF32},
    {"ffn_norm.weight", {2560}, kF32},
    {"post_ffw_norm.weight", {2560}, kF32},
    {"post_norm.weight", {2560}, kF32},
    {"attn_q_norm.weight", {256}, kF32},
    {"attn_k_norm.weight", {256}, kF32},
    {"inp_gate.weight", {256, 2560}, kF32},
    {"proj.weight", {2560, 256}, kF32},
    {"layer_output_scale.weight", {1}, kF32},
};

// E4B layer 5: full-attention (head_dim 512).
const std::vector<ExpectedTensor> kE4bLayer5Full = {
    {"attn_q.weight", {4096, 2560}, kQ8_0},
    {"attn_k.weight", {1024, 2560}, kQ8_0},
    {"attn_v.weight", {1024, 2560}, kQ8_0},
    {"attn_output.weight", {2560, 4096}, kQ8_0},
    {"ffn_gate.weight", {10240, 2560}, kQ8_0},
    {"ffn_up.weight", {10240, 2560}, kQ8_0},
    {"ffn_down.weight", {2560, 10240}, kQ8_0},
    {"attn_norm.weight", {2560}, kF32},
    {"post_attention_norm.weight", {2560}, kF32},
    {"ffn_norm.weight", {2560}, kF32},
    {"post_ffw_norm.weight", {2560}, kF32},
    {"post_norm.weight", {2560}, kF32},
    {"attn_q_norm.weight", {512}, kF32},
    {"attn_k_norm.weight", {512}, kF32},
    {"inp_gate.weight", {256, 2560}, kF32},
    {"proj.weight", {2560, 256}, kF32},
    {"layer_output_scale.weight", {1}, kF32},
};

// The six file-level tensors, both models -- verified-gguf-facts.md's
// "The six file-level tensors" section (E2B) and its "other shapes" section
// (E4B's token_embd/per_layer_token_embd/per_layer_model_proj). E4B's
// per_layer_proj_norm/output_norm/rope_freqs are derived the same way the
// per-layer tables above are (ple_dim and the full head_dim are both
// confirmed invariant across models); output_norm's width is hidden, the
// one file-level shape that must vary with the model by construction.
const std::vector<ExpectedTensor> kE2bFileLevel = {
    {"token_embd.weight", {262144, 1536}, kQ8_0},
    {"per_layer_token_embd.weight", {262144, 8960}, kQ8_0},
    // BF16, AND THAT IS A MEASUREMENT, not a guess: ggml type 30 on both
    // rows, the ONE BF16 tensor in either file (the reference driver's
    // `floats()` says so in as many words, and a re-dump of both real GGUFs
    // on 2026-09-22 confirms a type histogram of 353 F32 / 247 Q8_0 / 1
    // BF16 on E2B). It was kQ8_0 here, which made every Gemma 4 test in this
    // project agree with a file no vendor ships -- and, because
    // flm::rai::GgufFile could not size a type-30 tensor at all, was the
    // reason `Open` threw on both real files before any Gemma 4 code ran.
    {"per_layer_model_proj.weight", {8960, 1536}, kBf16},
    {"per_layer_proj_norm.weight", {256}, kF32},
    {"output_norm.weight", {1536}, kF32},
    {"rope_freqs.weight", {256}, kF32},
};

const std::vector<ExpectedTensor> kE4bFileLevel = {
    {"token_embd.weight", {262144, 2560}, kQ8_0},
    {"per_layer_token_embd.weight", {262144, 10752}, kQ8_0},
    {"per_layer_model_proj.weight", {10752, 2560}, kBf16},   // see the E2B table
    {"per_layer_proj_norm.weight", {256}, kF32},
    {"output_norm.weight", {2560}, kF32},
    {"rope_freqs.weight", {256}, kF32},
};

// Checks both shape AND type in one call, and -- because it hands off to
// flm::rai::GgufFile's own RequireQ8/RequireF32 rather than re-implementing
// the comparison -- inherits that parser's exact failure message: the
// tensor's name, its actual shape/type, and what was expected. A renamed
// tensor fails at "missing" (Impl::Tensor's own Fail()); a reshaped one
// fails naming the actual vs. expected shape.
void RequireTensor(const flm::rai::GgufFile& file, const std::string& full_name,
                   const ExpectedTensor& expected) {
    const std::vector<std::int64_t> shape(expected.shape.begin(), expected.shape.end());
    if (expected.type == kQ8_0) {
        file.RequireQ8(full_name, shape);
    } else if (expected.type == kF32) {
        file.RequireF32(full_name, shape);
    } else if (expected.type == kBf16) {
        file.RequireBf16(full_name, shape);
    } else {
        throw std::runtime_error("test bug: unknown expected type for " + full_name);
    }
}

void RequireLayerMatchesTable(const flm::rai::GgufFile& file, std::int64_t layer_index,
                              const std::vector<ExpectedTensor>& expected) {
    for (const auto& tensor : expected) {
        RequireTensor(file, "blk." + std::to_string(layer_index) + "." + std::string(tensor.name),
                     tensor);
    }
}

void RequireFileLevelMatchesTable(const flm::rai::GgufFile& file,
                                  const std::vector<ExpectedTensor>& expected) {
    for (const auto& tensor : expected) RequireTensor(file, std::string(tensor.name), tensor);
    TEST_REQUIRE(!file.HasTensor("output.weight"));
}

// `per_layer_model_proj.weight` is the one BF16 tensor in a Gemma 4 file,
// and asserting it HERE, on its own, rather than only inside the two big
// tables, is deliberate: this is the tensor whose type the container reader
// could not size at all, so "the fixture ships the type the real file does"
// and "GgufFile can read that type" are the two halves of the same defect
// and each deserves to fail by name.
void TestPerLayerModelProjIsBf16OnBothRowsLikeTheRealFiles() {
    for (const auto& options : {gemma4_fixture::E2bOptions(),
                                gemma4_fixture::E4bOptions()}) {
        const auto file = flm::rai::GgufFile::Open(gemma4_fixture::Write(options));
        const auto hidden = options.hidden;
        const auto ple_all = options.layers * 256;
        // Shape in GgufFile's own [OUT, IN] convention -- the file stores
        // this one as [hidden, layers * ple_dim].
        const std::vector<std::int64_t> shape{ple_all, hidden};
        const auto view = file->RequireBf16("per_layer_model_proj.weight", shape);
        TEST_REQUIRE(view.ggml_type == gemma4_fixture::kBf16);
        TEST_REQUIRE(view.values.size() ==
                     static_cast<std::size_t>(ple_all) * static_cast<std::size_t>(hidden));
        // Asked for as either of the other two types, it must be refused
        // naming BF16 -- so a fixture that silently reverted to Q8_0 could
        // not pass this by producing a same-sized payload.
        RequireContains(RequireThrows([&] { file->RequireQ8("per_layer_model_proj.weight", shape); }),
                        "BF16");
        RequireContains(RequireThrows([&] { file->RequireF32("per_layer_model_proj.weight", shape); }),
                        "BF16");
    }
}

void TestE2bPinsTheCompleteTensorTable() {
    const auto path = gemma4_fixture::Write(gemma4_fixture::E2bOptions());
    const auto file = flm::rai::GgufFile::Open(path);
    RequireFileLevelMatchesTable(*file, kE2bFileLevel);
    RequireLayerMatchesTable(*file, 0, kE2bLayer0Sliding);    // sliding, low FFN
    RequireLayerMatchesTable(*file, 4, kE2bLayer4Full);       // full,    low FFN
    RequireLayerMatchesTable(*file, 15, kE2bLayer15Sliding);  // sliding, high FFN
    RequireLayerMatchesTable(*file, 19, kE2bLayer19Full);     // full,    high FFN
}

void TestE4bPinsTheCompleteTensorTable() {
    const auto path = gemma4_fixture::Write(gemma4_fixture::E4bOptions());
    const auto file = flm::rai::GgufFile::Open(path);
    RequireFileLevelMatchesTable(*file, kE4bFileLevel);
    RequireLayerMatchesTable(*file, 0, kE4bLayer0Sliding);   // sliding
    RequireLayerMatchesTable(*file, 5, kE4bLayer5Full);      // full
}

// verified-gguf-facts.md recorded a real name-set diff between layer 0 and
// layer 4 of the real E2B file ("only on layer 0: [], only on layer 4: []")
// -- the two attention geometries differ ONLY in widths, never in which
// tensors exist. Check that directly, on an adjacent full/sliding pair,
// rather than leaving it as something the per-layer tables above only
// demonstrate as a side effect of reusing the same 17 names.
void TestFullAndSlidingLayersExposeTheSameTensorNames() {
    const auto path = gemma4_fixture::Write(gemma4_fixture::E2bOptions());
    const auto file = flm::rai::GgufFile::Open(path);
    for (const auto& tensor : kE2bLayer0Sliding) {
        const auto suffix = "." + std::string(tensor.name);
        TEST_REQUIRE(file->HasTensor("blk.4" + suffix));   // full-attention layer
        TEST_REQUIRE(file->HasTensor("blk.5" + suffix));   // sliding layer
    }
}
// -- fixture hygiene -------------------------------------------------------
//
// These two are about the fixture's FOOTPRINT, not about Gemma 4. They live
// here because this file already owns the fixture's own self-tests, and
// because a full E2B fixture is ~4.9 GB and a full E4B one ~8.2 GB: one run
// of this binary alone writes six of them. Until this round every one of
// those files survived the whole process and, on a FAILING run, survived the
// process too -- test_support.hpp's RunTest forces the process down with
// std::_Exit(1), which bypasses the static destructor the old cleanup hung
// off. A day of that filled a 1.9 TB volume to 99% with 240 files.

void TestWritingAFixtureRemovesTheOnesWrittenBeforeIt() {
    gemma4_fixture::Builder builder;
    builder.SetMetadata("general.architecture", std::string("gemma4"));
    builder.AddTensor("token_embd.weight", {32, 32}, gemma4_fixture::kF32);

    const auto first = builder.Write("hygiene-a");
    TEST_REQUIRE(std::filesystem::exists(first));

    const auto second = builder.Write("hygiene-b");
    // The earlier generation is gone and the latest one is kept -- kept
    // deliberately, so a failing run still leaves exactly one artifact to
    // look at rather than none or two hundred.
    TEST_REQUIRE(!std::filesystem::exists(first));
    TEST_REQUIRE(std::filesystem::exists(second));
}

void TestStaleFixturesFromADeadRunAreSweptButALiveRunsAreNot() {
    const auto temp = std::filesystem::temp_directory_path();
    const auto own_pid = gemma4_fixture::detail::CurrentProcessId();
    // 4294967295 is not a pid any process here has; it stands in for the run
    // that died holding this file.
    const auto stale = temp / "flm_gemma4_hygiene-stale_4294967295_1.gguf";
    const auto live = temp / ("flm_gemma4_hygiene-live_" +
                              std::to_string(own_pid) + "_1.gguf");
    { std::ofstream(stale, std::ios::binary) << 'x'; }
    { std::ofstream(live, std::ios::binary) << 'x'; }
    TEST_REQUIRE(std::filesystem::exists(stale));

    const auto cutoff = std::filesystem::last_write_time(stale) +
                        std::chrono::seconds(1);
    const auto removed =
        gemma4_fixture::detail::SweepStaleTempFixtures(cutoff, own_pid);

    TEST_REQUIRE(removed >= 1);
    TEST_REQUIRE(!std::filesystem::exists(stale));
    // OUR OWN pid is never swept, whatever its age. That is what keeps a
    // concurrent `ctest -j` sibling's in-flight fixture safe from us: the
    // pid is in the filename precisely so this check can be made.
    TEST_REQUIRE(std::filesystem::exists(live));
    std::error_code ignored;
    std::filesystem::remove(live, ignored);
}
}  // namespace

int main() {
#define RUN_TEST(name) RunTest(&name, #name)
    RUN_TEST(TestBothShippedRowsAreFound);
    RUN_TEST(TestAnUnknownShapeAsksForARowRatherThanGuessing);
    RUN_TEST(TestE2bFixtureMatchesTheRealFilesShapeAndTensorCount);
    RUN_TEST(TestE4bFixtureMatchesTheRealFilesShapeAndTensorCount);
    RUN_TEST(TestFeedForwardLengthMetadataIsSettableIndependentlyOfTensorShapes);
    RUN_TEST(TestPerLayerModelProjIsBf16OnBothRowsLikeTheRealFiles);
    RUN_TEST(TestE2bPinsTheCompleteTensorTable);
    RUN_TEST(TestE4bPinsTheCompleteTensorTable);
    RUN_TEST(TestFullAndSlidingLayersExposeTheSameTensorNames);
    RUN_TEST(TestWritingAFixtureRemovesTheOnesWrittenBeforeIt);
    RUN_TEST(TestStaleFixturesFromADeadRunAreSweptButALiveRunsAreNot);
#undef RUN_TEST
}
