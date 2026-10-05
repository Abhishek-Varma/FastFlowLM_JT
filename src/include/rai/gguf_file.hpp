/// \file gguf_file.hpp
/// \brief A memory-mapped GGUF v3 file, for the rai engines that read one
/// \note Only what a corelib engine needs: the tensor directory, as spans over
///       the mapping, and the scalar metadata a caller asks to keep. Arrays are
///       walked and skipped (only their element count is kept), which is what
///       keeps opening a file with a 150k-entry vocabulary cheap.
#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace flm::rai {

inline constexpr std::uint32_t kGgmlTypeF32 = 0;
inline constexpr std::uint32_t kGgmlTypeQ8_0 = 8;
inline constexpr std::uint32_t kGgmlTypeBF16 = 30;

/// \brief a tensor as the file stores it
/// \note `shape` is outermost-first -- the reverse of GGUF's own dimension
///       order -- so a projection from k to n reads [n, k].
struct GgufTensor {
    std::string_view name;
    std::span<const std::byte> bytes;
    std::vector<std::int64_t> shape;
    std::uint32_t ggml_type;
};

/// \brief an F32 tensor, viewed as floats in place
struct GgufFloatTensor {
    std::string_view name;
    std::span<const float> values;
    std::vector<std::int64_t> shape;
};

/// \brief a BF16 tensor, viewed in place as raw 16-bit patterns
struct GgufBf16Tensor {
    std::string_view name;
    std::span<const std::uint16_t> values;
    std::vector<std::int64_t> shape;
    std::uint32_t ggml_type;
};

class GgufFile final {
public:
    /// \brief which metadata keys to keep; everything else is parsed and dropped
    using RetainKey = std::function<bool(std::string_view)>;

    /// \brief map and index a GGUF v3 file
    /// \param retain which keys to keep; empty keeps every key
    /// \throws std::runtime_error naming the field when the file is malformed:
    ///         bad magic or version, ranges outside the file, misaligned or
    ///         overlapping tensors, duplicate names
    static std::shared_ptr<GgufFile> Open(const std::filesystem::path& path,
                                          const RetainKey& retain = {});
    ~GgufFile();

    const std::filesystem::path& Path() const;
    bool HasTensor(std::string_view name) const;
    bool HasMetadata(std::string_view key) const;

    /// \brief a Q8_0 tensor of exactly this shape
    GgufTensor RequireQ8(std::string_view name,
                         std::span<const std::int64_t> shape) const;
    /// \brief an F32 tensor of exactly this shape
    GgufFloatTensor RequireF32(std::string_view name,
                               std::span<const std::int64_t> shape) const;
    /// \brief a BF16 tensor of exactly this shape
    GgufBf16Tensor RequireBf16(std::string_view name,
                               std::span<const std::int64_t> shape) const;
    /// \brief whatever the file holds under this name, unchecked
    GgufTensor Tensor(std::string_view name) const;
    std::size_t TensorCount() const noexcept;

    std::uint64_t Unsigned(std::string_view key) const;
    double Number(std::string_view key) const;
    bool Boolean(std::string_view key) const;
    std::string String(std::string_view key) const;
    /// \brief the element count of an array-valued key
    std::uint64_t ArrayCount(std::string_view key) const;
    /// \brief the elements of an integer array, widened
    /// \throws when the elements are not integers or one is negative
    std::vector<std::uint64_t> UnsignedArray(std::string_view key) const;

private:
    struct Impl;
    explicit GgufFile(std::unique_ptr<Impl> impl);
    std::unique_ptr<Impl> impl_;
};

/// \brief "field: actual X, expected Y", the shape every rai load error takes
[[noreturn]] void FailField(std::string_view field, std::string actual,
                            std::string expected);

}  // namespace flm::rai
