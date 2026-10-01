#pragma once

#include <ryzenai/corelib.h>

#include <cstdint>
#include <initializer_list>

namespace flm::corelib {

/// \brief a component that lives in caller memory
inline ryzenai_corelib_weights_source MemorySource(
    ryzenai_corelib_weights_role role, const void* data, std::uint64_t bytes,
    int data_type, std::initializer_list<std::int64_t> dims) {
    ryzenai_corelib_weights_source source{};
    source.data = data;
    source.role = role;
    source.size = bytes;
    source.data_type = data_type;
    source.rank = static_cast<std::uint32_t>(dims.size());
    std::uint32_t axis = 0;
    for (const auto dim : dims) source.dims[axis++] = dim;
    return source;
}

/// \brief a Q8_0 block stream whose weight shape is [rows, cols]
inline ryzenai_corelib_weights_source GgufQ8(ryzenai_corelib_weights_role role,
                                             const void* data, std::uint64_t bytes,
                                             std::int64_t rows, std::int64_t cols) {
    return MemorySource(role, data, bytes, ryzenai_corelib_weights_data_type_gguf_q8_0,
                        {rows, cols});
}

/// \brief `count` BF16 values
inline ryzenai_corelib_weights_source Bf16(ryzenai_corelib_weights_role role,
                                           const void* data, std::uint64_t count) {
    return MemorySource(role, data, count * sizeof(std::uint16_t),
                        ryzenai_corelib_weights_data_type_onnx_bfloat16,
                        {static_cast<std::int64_t>(count)});
}

/// \brief `count` FP32 values, with the shape the packer checks
inline ryzenai_corelib_weights_source Fp32(ryzenai_corelib_weights_role role,
                                           const void* data, std::uint64_t count,
                                           std::initializer_list<std::int64_t> dims) {
    return MemorySource(role, data, count * sizeof(float),
                        ryzenai_corelib_weights_data_type_onnx_float, dims);
}

/// \brief `count` FP16 values
inline ryzenai_corelib_weights_source Fp16(ryzenai_corelib_weights_role role,
                                           const void* data, std::uint64_t count,
                                           std::initializer_list<std::int64_t> dims) {
    return MemorySource(role, data, count * sizeof(std::uint16_t),
                        ryzenai_corelib_weights_data_type_onnx_float16, dims);
}

/// \brief corelib's packed bytes for one weight set, at a slice of a file
/// \note The whole source set when given; corelib maps the file rather than
///       holding caller memory.
inline ryzenai_corelib_weights_source PackedFile(const char* path, std::uint64_t offset,
                                                 std::uint64_t size) {
    ryzenai_corelib_weights_source source{};
    source.path = path;
    source.role = ryzenai_corelib_weights_role_packed;
    source.offset = offset;
    source.size = size;
    source.data_type = ryzenai_corelib_weights_data_type_native_packed;
    return source;
}

/// \brief packing options with a thread-count hint (0 lets corelib choose)
inline ryzenai_corelib_weights_options PackOptions(std::uint32_t threads) {
    ryzenai_corelib_weights_options options{};
    options.fast_packer_threads = threads;
    return options;
}

}  // namespace flm::corelib
