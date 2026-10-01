#pragma once

#include <ryzenai/corelib.h>

#include <cstdint>
#include <initializer_list>

namespace flm::corelib {

/// \brief a component that lives in caller memory
inline ryzenai_corelib_weights_source MemorySource(
    const void* data, std::uint64_t bytes, int data_type,
    std::initializer_list<std::int64_t> dims) {
    ryzenai_corelib_weights_source source{};
    source.data = data;
    source.size = bytes;
    source.data_type = data_type;
    source.rank = static_cast<std::uint32_t>(dims.size());
    std::uint32_t axis = 0;
    for (const auto dim : dims) source.dims[axis++] = dim;
    return source;
}

/// \brief a Q8_0 block stream whose weight shape is [rows, cols]
inline ryzenai_corelib_weights_source GgufQ8(const void* data, std::uint64_t bytes,
                                             std::int64_t rows, std::int64_t cols) {
    return MemorySource(data, bytes, ryzenai_corelib_weights_data_type_gguf_q8_0,
                        {rows, cols});
}

/// \brief `count` BF16 values
inline ryzenai_corelib_weights_source Bf16(const void* data, std::uint64_t count) {
    return MemorySource(data, count * sizeof(std::uint16_t),
                        ryzenai_corelib_weights_data_type_onnx_bfloat16,
                        {static_cast<std::int64_t>(count)});
}

/// \brief `count` FP32 values, with the shape the packer checks
inline ryzenai_corelib_weights_source Fp32(const void* data, std::uint64_t count,
                                           std::initializer_list<std::int64_t> dims) {
    return MemorySource(data, count * sizeof(float),
                        ryzenai_corelib_weights_data_type_onnx_float, dims);
}

/// \brief `count` FP16 values
inline ryzenai_corelib_weights_source Fp16(const void* data, std::uint64_t count,
                                           std::initializer_list<std::int64_t> dims) {
    return MemorySource(data, count * sizeof(std::uint16_t),
                        ryzenai_corelib_weights_data_type_onnx_float16, dims);
}

/// \brief packed bytes already in a file
inline ryzenai_corelib_packed_weights_source PackedFile(const char* path,
                                                        std::uint64_t offset,
                                                        std::uint64_t size) {
    ryzenai_corelib_packed_weights_source source{};
    source.path = path;
    source.offset = offset;
    source.size = size;
    source.memory = ryzenai_corelib_weights_memory_copy;
    return source;
}

}  // namespace flm::corelib
