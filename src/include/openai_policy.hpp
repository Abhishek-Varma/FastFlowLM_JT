#pragma once

#include <algorithm>
#include <cstddef>
#include <string>
#include <vector>

#include "AutoModel/automodel.hpp"

// content type
enum class openai_content_type {
    CONTENT,
    REASONING,
    TOOL
};

// reasoning tags
struct reasoning_tags {
    std::string start_tag;
    std::string end_tag;
};

// response struct
struct openai_content {
    std::string text;
    openai_content_type type;
};



class openai_policy {
public:
    explicit openai_policy(chat_template_type_t template_type, bool start_in_reasoning = false);

    static reasoning_tags get_reasoning_tags(chat_template_type_t template_type);

    std::vector<openai_content> parse(const std::string& text);
    std::vector<openai_content> flush();
    void reset(bool start_in_reasoning = false);

    bool in_reasoning_part() const;

private:
    static void append(std::vector<openai_content>& parts,
                       openai_content_type type,
                       const std::string& text);

    static std::size_t partial_tag_length(const std::string& buffer, const std::string& tag);

    openai_content_type current_type() const;
    const std::string& pending_tag() const;

    chat_template_type_t template_type;
    reasoning_tags tags;
    std::string buffer;
    bool in_reasoning;
};

inline openai_policy::openai_policy(chat_template_type_t template_type, bool start_in_reasoning)
    : template_type(template_type),
      tags(get_reasoning_tags(template_type)),
      in_reasoning(start_in_reasoning) {
    buffer.reserve(256);
}

inline reasoning_tags openai_policy::get_reasoning_tags(chat_template_type_t template_type) {
    switch (template_type) {
        case chat_template_type_t::chat_ml:
            return reasoning_tags{"<think>", "</think>"};
        case chat_template_type_t::harmony:
            return reasoning_tags{"<|start|>assistant<|channel|>analysis<|message|>",
                                  "<|end|><|start|>assistant<|channel|>final<|message|>"};
        case chat_template_type_t::gemma4:
            return reasoning_tags{"<|channel>thought", "<channel|>"};
        case chat_template_type_t::unknown_template:
            return reasoning_tags{"", ""};
    }
    return reasoning_tags{"", ""};
}

inline std::vector<openai_content> openai_policy::parse(const std::string& text) {
    std::vector<openai_content> parts;
    // carry over whatever was held back last call, a tag may straddle two calls
    buffer += text;

    // consume every complete tag sitting in the buffer, one call can cross
    // several boundaries; pending_tag() is the start tag while we are outside
    // reasoning and the end tag while we are inside it
    while (!pending_tag().empty()) {
        const std::string& tag = pending_tag();
        std::size_t pos = buffer.find(tag);
        if (pos == std::string::npos) {
            // no boundary yet, the rest is handled below
            break;
        }
        // everything before the tag belongs to the part we are leaving
        append(parts, current_type(), buffer.substr(0, pos));
        // drop the tag itself, it is a marker and never reaches the client
        buffer.erase(0, pos + tag.size());
        in_reasoning = !in_reasoning;
    }

    // the tail may be the beginning of the next tag, hold that many bytes back
    // so a split marker is not emitted as visible text
    std::size_t held = partial_tag_length(buffer, pending_tag());
    if (buffer.size() > held) {
        std::size_t emit_len = buffer.size() - held;
        append(parts, current_type(), buffer.substr(0, emit_len));
        buffer.erase(0, emit_len);
    }

    // whatever stays in buffer is a partial tag, flush() releases it at end of stream
    return parts;
}

inline std::vector<openai_content> openai_policy::flush() {
    std::vector<openai_content> parts;
    append(parts, current_type(), buffer);
    buffer.clear();
    return parts;
}

inline void openai_policy::reset(bool start_in_reasoning) {
    buffer.clear();
    in_reasoning = start_in_reasoning;
}

inline bool openai_policy::in_reasoning_part() const {
    return in_reasoning;
}

inline void openai_policy::append(std::vector<openai_content>& parts,
                                  openai_content_type type,
                                  const std::string& text) {
    if (text.empty()) {
        return;
    }
    if (!parts.empty() && parts.back().type == type) {
        parts.back().text += text;
        return;
    }
    parts.push_back(openai_content{text, type});
}

inline std::size_t openai_policy::partial_tag_length(const std::string& buffer, const std::string& tag) {
    if (tag.empty() || buffer.empty()) {
        return 0;
    }
    std::size_t max_len = std::min(buffer.size(), tag.size() - 1);
    for (std::size_t len = max_len; len > 0; --len) {
        if (buffer.compare(buffer.size() - len, len, tag, 0, len) == 0) {
            return len;
        }
    }
    return 0;
}

inline openai_content_type openai_policy::current_type() const {
    return in_reasoning ? openai_content_type::REASONING : openai_content_type::CONTENT;
}

inline const std::string& openai_policy::pending_tag() const {
    return in_reasoning ? tags.end_tag : tags.start_tag;
}
