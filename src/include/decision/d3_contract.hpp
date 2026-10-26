/// \file d3_contract.hpp
/// \brief The d3 (Decision 3.0) System One contract: request validation, the prompt the model reads,
///        the readout, and the answer objects. A byte-exact port of the upstream package's
///        d3_format.py, d3_runtime.py (normalize_question, prepare, product_answer, respond) and
///        d3_server.py (request checks) at vllm-sr/d3-lite v3.1.3.
/// \note  No NPU code here: the engine returns one hidden row per question and everything else is
///        this file. src/test/d3_contract checks it against the frozen goldens and fuzz requests.
#pragma once

#include <functional>
#include <map>
#include <utility>
#include <stdexcept>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "decision/py_json.hpp"

namespace flm::d3 {

using ojson = nlohmann::ordered_json;

constexpr int MAX_OPTIONS = 255;
constexpr const char* FORMAT_ID = "d3-code-readout-v1";
extern const char* const SYSTEM_PROMPT;

/// A problem with one question (`invalid_question`) or with a model output (`invalid_model_output`);
/// the message is upstream's ValueError text.
struct ContractError : std::runtime_error {
    using std::runtime_error::runtime_error;
};

/// A request the server answers with 422 `{"detail": detail}` before running anything.
struct RequestError : std::runtime_error {
    ojson detail;
    explicit RequestError(ojson d) : std::runtime_error(d.is_string() ? d.get<std::string>() : d.dump()), detail(std::move(d)) {}
};

/// One validated question (d3_runtime.Question).
struct Question {
    std::string kind;                       //!< choice | noul | score
    std::vector<std::string> keys;          //!< answer keys, in option order ("0".."n-1" for score)
    std::vector<std::string> option_texts;  //!< the text after each code in the prompt
    std::vector<PyValue> levels;            //!< score: the original levels (legend)
    PyValue instructions;                   //!< as given (null when absent)
};

/// `describe`: strings verbatim, anything else as `json.dumps(v, ensure_ascii=False)`.
std::string describe(const PyValue& v);

/// `normalize_question`; throws ContractError with upstream's message.
Question normalize_question(const PyValue& question);

/// The 255 candidate codes in preference order (A..Z, AA..ZZ).
std::vector<std::string> candidate_codes();

/// `d3_format.user_prompt` (state may be null).
std::string user_prompt(const PyValue& state, const Question& q, const std::vector<std::string>& codes);

/// The chat text of `[system, user]` (or `[user]` when system is empty) with
/// `add_generation_prompt=True, enable_thinking=False`, as the package's chat_template.jinja renders
/// it. AutoDecisionModel cross-checks this against minja at load.
std::string chat_text(const std::string& system, const std::string& user);

/// The whole prompt of one question (`d3_format.render`).
std::string render(const PyValue& state, const Question& q, const std::vector<std::string>& codes);

// ---- images (d3_runtime: load_image strict, visual_tokens, image_messages) ----------------------
constexpr const char* IMAGE_TOKEN = "<|image_pad|>";
constexpr const char* VIDEO_TOKEN = "<|video_pad|>";
constexpr long long MAX_IMAGE_BYTES = 8000000;           //!< decoded bytes of one image
constexpr long long MAX_IMAGE_SOURCE_PIXELS = 16000000;  //!< width x height before the resize
constexpr int IMAGE_MIN_PIXELS = 65536;                  //!< resize bounds (the processor's size)
constexpr int IMAGE_MAX_PIXELS = 1638400;
constexpr int IMAGE_PATCH_SIZE = 16;
constexpr int IMAGE_MERGE_SIZE = 2;

/// One request image as the server reads it before decoding pixels: the file bytes, the format Pillow
/// would identify ("PNG", "JPEG", "WEBP") and the size in its header.
struct ImageFile {
    std::string data;
    std::string format;
    int width = 0;
    int height = 0;
};

/// `load_image(value, strict=True)` up to decoding: a data URL string of type png/jpeg/jpg/webp, valid
/// base64 within MAX_IMAGE_BYTES, a PNG/JPEG/WebP file of at most MAX_IMAGE_SOURCE_PIXELS. Throws
/// ContractError with upstream's message (the caller adds "images[i]: "). Decoding and the integrity
/// check are the application's.
ImageFile inspect_image(const PyValue& value);

/// The processor's `smart_resize` (factor = patch x merge): (height, width) of the resized image.
/// Throws ContractError("absolute aspect ratio must be smaller than 200, got <ratio>").
std::pair<int, int> smart_resize(int height, int width, int factor = IMAGE_PATCH_SIZE * IMAGE_MERGE_SIZE,
                                 int min_pixels = IMAGE_MIN_PIXELS, int max_pixels = IMAGE_MAX_PIXELS);

/// The prompt of one question with `n_images` image placeholders in front of the user text
/// (`image_messages` rendered by the chat template).
std::string render_image(const PyValue& state, const Question& q, const std::vector<std::string>& codes,
                         int n_images);

/// `d3_format.answer_codes`: the codes and token ids this tokenizer gives; throws ContractError if
/// fewer than 255 qualify.
void answer_codes(const std::function<std::vector<int>(const std::string&)>& encode,
                  std::vector<std::string>& codes, std::vector<int>& token_ids);

/// Masked code logits -> option probabilities, as the runtime computes them: fp32 hidden @ fp32
/// readout^T, codes >= count masked, / temperature, fp32 softmax, cut to count.
std::vector<float> readout_probabilities(const float* hidden, const float* readout, int hidden_size,
                                         int count, float temperature);

/// `product_answer`: the public answer object; throws ContractError (invalid_model_output) when the
/// probabilities are unusable.
ojson product_answer(const Question& q, const std::vector<double>& probabilities);

/// `{"type": <type as given>, "error": error, "message": message}`.
ojson error_answer(const ojson& type, const std::string& error, const std::string& message);

/// A PyValue as nlohmann JSON (for echoing request values in replies).
ojson to_ojson(const PyValue& v);

/// A System One request after validation and rendering (d3_runtime.Prepared, text and image paths).
struct Prepared {
    std::vector<std::string> keys;              //!< every question ID, request order
    std::map<std::string, Question> questions;  //!< valid questions
    std::map<std::string, std::string> texts;   //!< rendered prompts of valid questions
    std::map<std::string, std::vector<int>> sequences;  //!< token ids of runnable questions
    std::map<std::string, ojson> errors;        //!< per-question error objects
    bool has_images = false;
    bool has_videos = false;
    std::vector<ImageFile> images;              //!< the request's images, in order (shared by every question)
    std::vector<std::pair<int, int>> image_grids; //!< per image: (grid_h, grid_w) in 16-pixel patches
    std::vector<int> image_tokens;              //!< per image: input tokens, grid_h * grid_w / 4

    std::vector<std::string> runnable() const;
};

/// Server-level checks of a decoded body (d3_server.Decider.arguments + D3.prepare's request
/// checks) and per-question validation/rendering. Throws RequestError for what upstream answers 422.
/// Tokenization and the length limit are applied by `tokenize()`. Images are inspected in order, and
/// `decode` (when given) decodes each one right after its inspection, as upstream loads them one by
/// one: it throws ContractError with the message for "images[i]: ..." (integrity, decoding).
Prepared prepare(const PyValue& body, const std::vector<std::string>& codes,
                 const std::function<void(size_t, const ImageFile&)>& decode = nullptr);

/// Tokenize every rendered prompt; prompts over `max_length` (> 0) become max_length_exceeded. With
/// images, each `<|image_pad|>` becomes that image's run of input tokens (counted in the length).
void tokenize(Prepared& p, const std::function<std::vector<int>(const std::string&)>& encode, int max_length);

/// `invalid_questions` of a prepared request, or an empty object.
ojson invalid_questions(const Prepared& p);

/// `D3.respond`: `{"model", "answers", "usage"}` from per-question probabilities.
ojson respond(const Prepared& p, const std::map<std::string, std::vector<double>>& probabilities,
              const std::string& model_name);

}  // namespace flm::d3
