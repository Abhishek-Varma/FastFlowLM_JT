/// \file chat_continue.hpp
/// \brief Continue the final assistant message instead of starting a new turn
/// \note The request field continue_final_message (vLLM and Hugging Face): the
///       chat is rendered without a generation prompt and cut right after the
///       final message's text, so the model writes on from there. The cut follows
///       transformers' apply_chat_template(continue_final_message=True), so the
///       same template gives the same prompt here as there.
#pragma once

#include <string>

#include <nlohmann/json.hpp>

/// \brief Cut a rendered chat right after the text of its final message
/// \param rendered the chat rendered without a generation prompt
/// \param final_message the last message as the template was given it; its
///        content is a string or an array of typed parts, whose last text part counts
/// \return the rendered chat up to the end of the final message's text
/// \note Whitespace around the text is matched loosely, as transformers does:
///       templates often trim the content. Trailing whitespace is kept only when
///       the template rendered it.
/// \throws std::invalid_argument when the final message has no text, or the
///         template did not render that text as it is (some templates rewrite the
///         last assistant message, e.g. by moving a <think> block out of it)
std::string chat_continue_final_message(const std::string& rendered, const nlohmann::ordered_json& final_message);
