#include "models/qwen3/rai/aie_next/qwen3_rai_config.hpp"

#include <cmath>
#include <sstream>
#include <stdexcept>

namespace flm::qwen3 {
namespace {

bool Close(double a, double b) {
    return std::isfinite(a) && std::fabs(a - b) <= 1e-6 * std::fabs(b);
}

}  // namespace

const Qwen3Config& SelectQwen3Row(const Qwen3Shape& shape) {
    for (const auto& row : kQwen3Rows) {
        if (row.layers != shape.layers || row.hidden != shape.hidden ||
            row.q_heads != shape.q_heads || row.kv_heads != shape.kv_heads ||
            row.head_dim != shape.head_dim || row.intermediate != shape.intermediate)
            continue;
        // A GGUF stores these as float32, so 1e-6 reads back as 9.99999997e-07.
        if (!Close(shape.epsilon, row.epsilon))
            throw std::runtime_error("Qwen3-" + std::string(row.size) + " expects rms epsilon " +
                                     std::to_string(row.epsilon) + ", but the GGUF says " +
                                     std::to_string(shape.epsilon));
        if (!Close(shape.rope_theta, row.rope_theta))
            throw std::runtime_error("Qwen3-" + std::string(row.size) + " expects rope theta " +
                                     std::to_string(row.rope_theta) + ", but the GGUF says " +
                                     std::to_string(shape.rope_theta));
        return row;
    }
    std::ostringstream message;
    message << "no supported Qwen3 size matches this GGUF: layers=" << shape.layers
            << " hidden=" << shape.hidden << " heads=" << shape.q_heads << "/" << shape.kv_heads
            << " head_dim=" << shape.head_dim << " intermediate=" << shape.intermediate
            << " (the rai backend runs Qwen3 0.6B, 1.7B, 4B and 8B)";
    throw std::runtime_error(message.str());
}

}  // namespace flm::qwen3
