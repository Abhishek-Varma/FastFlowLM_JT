#include "rai/gguf_file.hpp"

#include "utils/file_access.hpp"

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <cerrno>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

#include <algorithm>
#include <cstring>
#include <limits>
#include <map>
#include <sstream>
#include <stdexcept>
#include <utility>
#include <variant>

namespace flm::rai {
namespace {
constexpr std::uint32_t kMagic = 0x46554747;
constexpr std::uint32_t kVersion = 3;
constexpr std::uint64_t kDefaultAlignment = 32;

std::uint64_t CheckedAdd(std::uint64_t a, std::uint64_t b, std::string_view field) {
    if (a > std::numeric_limits<std::uint64_t>::max() - b)
        throw std::runtime_error(std::string(field) + ": overflow in addition");
    return a + b;
}

std::uint64_t CheckedMultiply(std::uint64_t a, std::uint64_t b, std::string_view field) {
    if (a != 0 && b > std::numeric_limits<std::uint64_t>::max() / a)
        throw std::runtime_error(std::string(field) + ": overflow in product");
    return a * b;
}

std::uint64_t AlignUp(std::uint64_t value, std::uint64_t alignment) {
    return CheckedAdd(value, alignment - 1, "alignment") & ~(alignment - 1);
}

std::span<const std::byte> RequireRange(std::span<const std::byte> file,
                                        std::uint64_t offset, std::uint64_t length,
                                        std::string_view field) {
    const auto end = CheckedAdd(offset, length, field);
    if (end > file.size() || offset > std::numeric_limits<std::size_t>::max() ||
        length > std::numeric_limits<std::size_t>::max())
        FailField(field, "out-of-file range", "range within mapped file");
    return file.subspan(static_cast<std::size_t>(offset), static_cast<std::size_t>(length));
}

class Cursor {
public:
    explicit Cursor(std::span<const std::byte> file) : file_(file) {}

    template <typename T>
    T Read(std::string_view field) {
        const auto bytes = RequireRange(file_, offset_, sizeof(T), field);
        T value;
        std::memcpy(&value, bytes.data(), sizeof(T));
        offset_ = CheckedAdd(offset_, sizeof(T), field);
        return value;
    }

    std::string ReadString(std::string_view field) {
        const auto length = Read<std::uint64_t>(field);
        const auto bytes = RequireRange(file_, offset_, length, field);
        std::string value(reinterpret_cast<const char*>(bytes.data()), bytes.size());
        offset_ = CheckedAdd(offset_, length, field);
        return value;
    }

    void Skip(std::uint64_t length, std::string_view field) {
        RequireRange(file_, offset_, length, field);
        offset_ = CheckedAdd(offset_, length, field);
    }

    std::uint64_t offset() const noexcept { return offset_; }
    std::span<const std::byte> remaining() const {
        return file_.subspan(static_cast<std::size_t>(offset_));
    }

private:
    std::span<const std::byte> file_;
    std::uint64_t offset_{};
};

std::string MetadataTypeName(std::uint32_t type) {
    static constexpr const char* names[] = {"UINT8", "INT8", "UINT16", "INT16",
        "UINT32", "INT32", "FLOAT32", "BOOL", "STRING", "ARRAY", "UINT64",
        "INT64", "FLOAT64"};
    return type < std::size(names) ? names[type] : "unknown(" + std::to_string(type) + ")";
}

std::uint64_t FixedMetadataSize(std::uint32_t type) {
    switch (type) {
    case 0: case 1: case 7: return 1;
    case 2: case 3: return 2;
    case 4: case 5: case 6: return 4;
    case 10: case 11: case 12: return 8;
    default: return 0;
    }
}

bool IsIntegerMetadata(std::uint32_t type) {
    return type <= 5 || type == 10 || type == 11;
}

/// \note `values` is filled only for a retained integer array; every other
///       array keeps just its count.
struct ArrayInfo {
    std::uint32_t type;
    std::uint64_t count;
    std::vector<std::uint64_t> values;
    bool has_negative = false;
};
using MetadataValue = std::variant<std::monostate, std::uint64_t, std::int64_t,
                                   double, bool, std::string, ArrayInfo>;

MetadataValue ReadMetadataValue(Cursor& cursor, std::uint32_t type,
                                std::string_view field, bool retain) {
    switch (type) {
    case 0: { auto v = cursor.Read<std::uint8_t>(field); return retain ? MetadataValue(std::uint64_t(v)) : MetadataValue{}; }
    case 1: { auto v = cursor.Read<std::int8_t>(field); return retain ? MetadataValue(std::int64_t(v)) : MetadataValue{}; }
    case 2: { auto v = cursor.Read<std::uint16_t>(field); return retain ? MetadataValue(std::uint64_t(v)) : MetadataValue{}; }
    case 3: { auto v = cursor.Read<std::int16_t>(field); return retain ? MetadataValue(std::int64_t(v)) : MetadataValue{}; }
    case 4: { auto v = cursor.Read<std::uint32_t>(field); return retain ? MetadataValue(std::uint64_t(v)) : MetadataValue{}; }
    case 5: { auto v = cursor.Read<std::int32_t>(field); return retain ? MetadataValue(std::int64_t(v)) : MetadataValue{}; }
    case 6: { auto v = cursor.Read<float>(field); return retain ? MetadataValue(double(v)) : MetadataValue{}; }
    case 7: { auto v = cursor.Read<std::uint8_t>(field); if (v > 1) FailField(field, std::to_string(v), "GGUF boolean 0 or 1"); return retain ? MetadataValue(bool(v)) : MetadataValue{}; }
    case 8: { auto v = cursor.ReadString(field); return retain ? MetadataValue(std::move(v)) : MetadataValue{}; }
    case 9: {
        const std::string array_field = std::string(field) + " array";
        const auto element_type = cursor.Read<std::uint32_t>(array_field);
        const auto count = cursor.Read<std::uint64_t>(array_field);
        if (element_type == 9 || element_type > 12)
            FailField(array_field, MetadataTypeName(element_type), "a skippable GGUF array element type");
        ArrayInfo info{element_type, count};
        const auto fixed = FixedMetadataSize(element_type);
        if (retain && IsIntegerMetadata(element_type)) {
            const auto bytes = CheckedMultiply(count, fixed, array_field);
            RequireRange(cursor.remaining(), 0, bytes, array_field);
            info.values.reserve(static_cast<std::size_t>(count));
            for (std::uint64_t i = 0; i < count; ++i) {
                const auto element = ReadMetadataValue(cursor, element_type, array_field, true);
                if (const auto* u = std::get_if<std::uint64_t>(&element)) {
                    info.values.push_back(*u);
                } else {
                    const auto s = std::get<std::int64_t>(element);
                    if (s < 0) info.has_negative = true;
                    info.values.push_back(static_cast<std::uint64_t>(s));
                }
            }
        } else if (fixed != 0) {
            cursor.Skip(CheckedMultiply(count, fixed, array_field), array_field);
        } else {
            for (std::uint64_t i = 0; i < count; ++i)
                (void)ReadMetadataValue(cursor, element_type, array_field, false);
        }
        return retain ? MetadataValue(std::move(info)) : MetadataValue{};
    }
    case 10: { auto v = cursor.Read<std::uint64_t>(field); return retain ? MetadataValue(v) : MetadataValue{}; }
    case 11: { auto v = cursor.Read<std::int64_t>(field); return retain ? MetadataValue(v) : MetadataValue{}; }
    case 12: { auto v = cursor.Read<double>(field); return retain ? MetadataValue(v) : MetadataValue{}; }
    default:
        FailField(field, MetadataTypeName(type), "a supported metadata type");
    }
}

std::string ShapeText(std::span<const std::int64_t> shape) {
    std::ostringstream out;
    out << '[';
    for (std::size_t i = 0; i < shape.size(); ++i) {
        if (i) out << ',';
        out << shape[i];
    }
    return out.str() + ']';
}

std::string GgmlTypeName(std::uint32_t type) {
    if (type == kGgmlTypeF32) return "F32";
    if (type == kGgmlTypeQ8_0) return "Q8_0";
    if (type == kGgmlTypeBF16) return "BF16";
    return "GGML type " + std::to_string(type);
}

std::uint64_t ElementCount(std::span<const std::int64_t> shape, std::string_view field) {
    std::uint64_t result = 1;
    for (const auto dimension : shape) {
        if (dimension <= 0) FailField(field, std::to_string(dimension), "positive dimensions");
        result = CheckedMultiply(result, static_cast<std::uint64_t>(dimension), field);
    }
    return result;
}

/// \return the byte length, or 0 for a type this reader does not size
/// \note A file may carry tensors of types no rai engine reads (a K-quant
///       output head, say); those are indexed without a range check and
///       rejected only if something asks for them.
std::uint64_t TensorByteLength(std::uint32_t type, std::span<const std::int64_t> shape,
                               std::string_view field) {
    const auto elements = ElementCount(shape, field);
    if (type == kGgmlTypeF32) return CheckedMultiply(elements, 4, field);
    if (type == kGgmlTypeBF16) return CheckedMultiply(elements, 2, field);
    if (type == kGgmlTypeQ8_0) {
        if (elements % 32 != 0)
            FailField(field, std::to_string(elements) + " elements", "Q8_0 element count divisible by 32");
        return CheckedMultiply(elements / 32, 34, field);
    }
    return 0;
}
}  // namespace

void FailField(std::string_view field, std::string actual, std::string expected) {
    throw std::runtime_error(std::string(field) + ": actual " + actual +
                             ", expected " + expected);
}

struct GgufFile::Impl {
    struct TensorRecord {
        std::string name;
        std::span<const std::byte> bytes;
        std::vector<std::int64_t> shape;
        std::uint32_t type;
        std::uint64_t absolute_offset;
    };

    std::filesystem::path path;
#ifdef _WIN32
    HANDLE file = INVALID_HANDLE_VALUE;
    HANDLE mapping = nullptr;
#else
    int file = -1;
#endif
    const std::byte* data = nullptr;
    std::uint64_t size = 0;
    std::map<std::string, TensorRecord, std::less<>> tensors;
    std::map<std::string, MetadataValue, std::less<>> metadata;
    std::map<std::string, std::uint32_t, std::less<>> metadata_types;

    ~Impl() {
#ifdef _WIN32
        if (data) UnmapViewOfFile(data);
        if (mapping) CloseHandle(mapping);
        if (file != INVALID_HANDLE_VALUE) CloseHandle(file);
#else
        if (data)
            ::munmap(const_cast<std::byte*>(data), static_cast<std::size_t>(size));
        if (file >= 0) ::close(file);
#endif
    }

    std::span<const std::byte> bytes() const { return {data, static_cast<std::size_t>(size)}; }

    const TensorRecord& Record(std::string_view name) const {
        const auto it = tensors.find(name);
        if (it == tensors.end()) FailField(name, "missing", "present tensor");
        return it->second;
    }

    const MetadataValue& Value(std::string_view key, const char* expected) const {
        const auto it = metadata.find(key);
        if (it == metadata.end()) FailField(key, "missing", expected);
        return it->second;
    }

    [[noreturn]] void WrongType(std::string_view key, const char* expected) const {
        FailField(key, MetadataTypeName(metadata_types.find(key)->second), expected);
    }

    void CheckShape(const TensorRecord& tensor, std::uint32_t type,
                    std::span<const std::int64_t> expected) const {
        if (tensor.type != type) FailField(tensor.name, GgmlTypeName(tensor.type), GgmlTypeName(type));
        if (!std::equal(tensor.shape.begin(), tensor.shape.end(), expected.begin(), expected.end()))
            FailField(tensor.name, ShapeText(tensor.shape), ShapeText(expected));
        const auto length = TensorByteLength(type, expected, tensor.name);
        if (tensor.bytes.size() != length)
            FailField(tensor.name, std::to_string(tensor.bytes.size()) + " bytes",
                      std::to_string(length) + " bytes");
    }
};

GgufFile::GgufFile(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}
GgufFile::~GgufFile() = default;

std::shared_ptr<GgufFile> GgufFile::Open(const std::filesystem::path& path,
                                         const RetainKey& retain) {
    auto impl = std::make_unique<Impl>();
    impl->path = path;
    flm::file_access::ObserveOpen(path);
#ifdef _WIN32
    impl->file = CreateFileW(path.c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr,
                             OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
    if (impl->file == INVALID_HANDLE_VALUE)
        throw std::runtime_error("GGUF file " + path.string() + ": actual open failure " +
                                 std::to_string(GetLastError()) + ", expected readable file");
    LARGE_INTEGER size;
    if (!GetFileSizeEx(impl->file, &size) || size.QuadPart <= 0 ||
        static_cast<unsigned long long>(size.QuadPart) > std::numeric_limits<std::size_t>::max())
        FailField("GGUF file size", std::to_string(size.QuadPart), "positive mappable size");
    impl->size = static_cast<std::uint64_t>(size.QuadPart);
    impl->mapping = CreateFileMappingW(impl->file, nullptr, PAGE_READONLY, 0, 0, nullptr);
    if (!impl->mapping)
        throw std::runtime_error("GGUF mapping: actual CreateFileMappingW failure " +
                                 std::to_string(GetLastError()) + ", expected PAGE_READONLY mapping");
    impl->data = static_cast<const std::byte*>(MapViewOfFile(impl->mapping, FILE_MAP_READ, 0, 0, 0));
    if (!impl->data)
        throw std::runtime_error("GGUF mapping: actual MapViewOfFile failure " +
                                 std::to_string(GetLastError()) + ", expected FILE_MAP_READ view");
#else
    impl->file = ::open(path.c_str(), O_RDONLY | O_CLOEXEC);
    if (impl->file < 0)
        throw std::runtime_error("GGUF file " + path.string() + ": actual open failure " +
                                 std::to_string(errno) + ", expected readable file");
    struct ::stat status {};
    if (::fstat(impl->file, &status) != 0 || status.st_size <= 0 ||
        static_cast<unsigned long long>(status.st_size) >
            std::numeric_limits<std::size_t>::max())
        FailField("GGUF file size",
                  std::to_string(static_cast<long long>(status.st_size)),
                  "positive mappable size");
    impl->size = static_cast<std::uint64_t>(status.st_size);
    void* view = ::mmap(nullptr, static_cast<std::size_t>(impl->size), PROT_READ,
                        MAP_PRIVATE, impl->file, 0);
    if (view == MAP_FAILED)
        throw std::runtime_error("GGUF mapping: actual mmap failure " +
                                 std::to_string(errno) + ", expected PROT_READ view");
    impl->data = static_cast<const std::byte*>(view);
#endif

    const auto file = impl->bytes();
    Cursor cursor(file);
    if (cursor.Read<std::uint32_t>("GGUF header") != kMagic)
        FailField("GGUF magic", "mismatch", "0x46554747");
    const auto version = cursor.Read<std::uint32_t>("GGUF header");
    if (version != kVersion) FailField("GGUF version", std::to_string(version), "3");
    const auto tensor_count = cursor.Read<std::uint64_t>("tensor count");
    const auto metadata_count = cursor.Read<std::uint64_t>("metadata count");
    if (tensor_count > file.size() / 24)
        FailField("tensor count", std::to_string(tensor_count), "count fitting directory");
    if (metadata_count > file.size() / 12)
        FailField("metadata count", std::to_string(metadata_count), "count fitting metadata");

    for (std::uint64_t i = 0; i < metadata_count; ++i) {
        const auto key = cursor.ReadString("metadata key string");
        const auto type = cursor.Read<std::uint32_t>(key);
        const bool keep = key == "general.alignment" || !retain || retain(key);
        auto value = ReadMetadataValue(cursor, type, key, keep);
        if (keep) {
            if (!impl->metadata.emplace(key, std::move(value)).second)
                FailField(key, "duplicate metadata key", "unique metadata key");
            impl->metadata_types.emplace(key, type);
        }
    }

    std::uint64_t alignment = kDefaultAlignment;
    if (impl->metadata.contains("general.alignment")) {
        const auto* value = std::get_if<std::uint64_t>(&impl->metadata.at("general.alignment"));
        if (!value) impl->WrongType("general.alignment", "unsigned integer metadata");
        alignment = *value;
    }
    if (alignment == 0 || (alignment & (alignment - 1)) != 0)
        FailField("general.alignment", std::to_string(alignment), "a non-zero power of two");

    struct DirectoryTensor {
        std::string name;
        std::vector<std::int64_t> shape;
        std::uint32_t type;
        std::uint64_t relative_offset;
        std::uint64_t length;
    };
    std::vector<DirectoryTensor> directory;
    directory.reserve(static_cast<std::size_t>(tensor_count));
    for (std::uint64_t i = 0; i < tensor_count; ++i) {
        auto name = cursor.ReadString("tensor directory name");
        const auto dimension_count = cursor.Read<std::uint32_t>("tensor directory dimensions");
        if (dimension_count == 0 || dimension_count > 4)
            FailField(name, std::to_string(dimension_count), "1..4 tensor dimensions");
        std::vector<std::int64_t> shape;
        shape.reserve(dimension_count);
        for (std::uint32_t d = 0; d < dimension_count; ++d) {
            const auto dimension = cursor.Read<std::uint64_t>("tensor directory dimension");
            if (dimension > static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max()))
                FailField(name, std::to_string(dimension), "dimension fitting int64");
            shape.push_back(static_cast<std::int64_t>(dimension));
        }
        std::reverse(shape.begin(), shape.end());
        const auto type = cursor.Read<std::uint32_t>("tensor directory type");
        const auto offset = cursor.Read<std::uint64_t>("tensor directory offset");
        const auto length = TensorByteLength(type, shape, name);
        directory.push_back({std::move(name), std::move(shape), type, offset, length});
    }

    const auto data_start = AlignUp(cursor.offset(), alignment);
    struct Range { std::uint64_t begin, end; std::string name; };
    std::vector<Range> ranges;
    ranges.reserve(directory.size());
    for (auto& tensor : directory) {
        if (tensor.relative_offset % alignment != 0)
            FailField(tensor.name, std::to_string(tensor.relative_offset),
                      "offset aligned to " + std::to_string(alignment));
        const auto absolute = CheckedAdd(data_start, tensor.relative_offset, tensor.name);
        std::span<const std::byte> bytes;
        if (tensor.length != 0) {
            bytes = RequireRange(file, absolute, tensor.length, tensor.name + " range");
            ranges.push_back({absolute, CheckedAdd(absolute, tensor.length, tensor.name), tensor.name});
        }
        auto [it, inserted] = impl->tensors.emplace(tensor.name,
            Impl::TensorRecord{tensor.name, bytes, std::move(tensor.shape), tensor.type, absolute});
        if (!inserted) FailField(tensor.name, "duplicate tensor name", "unique tensor name");
    }
    std::sort(ranges.begin(), ranges.end(), [](const Range& a, const Range& b) { return a.begin < b.begin; });
    for (std::size_t i = 1; i < ranges.size(); ++i) {
        if (ranges[i].begin < ranges[i - 1].end)
            FailField(ranges[i].name, "overlap with " + ranges[i - 1].name, "non-overlapping tensor range");
    }
    return std::shared_ptr<GgufFile>(new GgufFile(std::move(impl)));
}

const std::filesystem::path& GgufFile::Path() const { return impl_->path; }

bool GgufFile::HasTensor(std::string_view name) const { return impl_->tensors.contains(name); }

bool GgufFile::HasMetadata(std::string_view key) const { return impl_->metadata.contains(key); }

GgufTensor GgufFile::RequireQ8(std::string_view name, std::span<const std::int64_t> shape) const {
    const auto& tensor = impl_->Record(name);
    impl_->CheckShape(tensor, kGgmlTypeQ8_0, shape);
    return {tensor.name, tensor.bytes, tensor.shape, tensor.type};
}

GgufFloatTensor GgufFile::RequireF32(std::string_view name, std::span<const std::int64_t> shape) const {
    const auto& tensor = impl_->Record(name);
    impl_->CheckShape(tensor, kGgmlTypeF32, shape);
    const auto address = reinterpret_cast<std::uintptr_t>(tensor.bytes.data());
    if (tensor.absolute_offset % alignof(float) != 0 || address % alignof(float) != 0)
        FailField(name, "address/offset not aligned", "alignment 4");
    return {tensor.name,
            {reinterpret_cast<const float*>(tensor.bytes.data()), tensor.bytes.size() / sizeof(float)},
            tensor.shape};
}

GgufBf16Tensor GgufFile::RequireBf16(std::string_view name, std::span<const std::int64_t> shape) const {
    const auto& tensor = impl_->Record(name);
    impl_->CheckShape(tensor, kGgmlTypeBF16, shape);
    const auto address = reinterpret_cast<std::uintptr_t>(tensor.bytes.data());
    if (tensor.absolute_offset % alignof(std::uint16_t) != 0 || address % alignof(std::uint16_t) != 0)
        FailField(name, "address/offset not aligned", "alignment 2");
    return {tensor.name,
            {reinterpret_cast<const std::uint16_t*>(tensor.bytes.data()),
             tensor.bytes.size() / sizeof(std::uint16_t)},
            tensor.shape, tensor.type};
}

GgufTensor GgufFile::Tensor(std::string_view name) const {
    const auto& tensor = impl_->Record(name);
    return {tensor.name, tensor.bytes, tensor.shape, tensor.type};
}

std::size_t GgufFile::TensorCount() const noexcept { return impl_->tensors.size(); }

std::uint64_t GgufFile::Unsigned(std::string_view key) const {
    const auto& value = impl_->Value(key, "unsigned integer metadata");
    if (const auto* v = std::get_if<std::uint64_t>(&value)) return *v;
    if (const auto* v = std::get_if<std::int64_t>(&value); v && *v >= 0)
        return static_cast<std::uint64_t>(*v);
    impl_->WrongType(key, "unsigned integer metadata");
}

double GgufFile::Number(std::string_view key) const {
    const auto& value = impl_->Value(key, "floating-point metadata");
    if (const auto* v = std::get_if<double>(&value)) return *v;
    impl_->WrongType(key, "floating-point metadata");
}

bool GgufFile::Boolean(std::string_view key) const {
    const auto& value = impl_->Value(key, "boolean metadata");
    if (const auto* v = std::get_if<bool>(&value)) return *v;
    impl_->WrongType(key, "boolean metadata");
}

std::string GgufFile::String(std::string_view key) const {
    const auto& value = impl_->Value(key, "string metadata");
    if (const auto* v = std::get_if<std::string>(&value)) return *v;
    impl_->WrongType(key, "string metadata");
}

std::uint64_t GgufFile::ArrayCount(std::string_view key) const {
    const auto& value = impl_->Value(key, "array metadata");
    if (const auto* v = std::get_if<ArrayInfo>(&value)) return v->count;
    impl_->WrongType(key, "array metadata");
}

std::vector<std::uint64_t> GgufFile::UnsignedArray(std::string_view key) const {
    const auto& value = impl_->Value(key, "unsigned integer array metadata");
    const auto* array = std::get_if<ArrayInfo>(&value);
    if (!array) impl_->WrongType(key, "unsigned integer array metadata");
    if (!IsIntegerMetadata(array->type))
        FailField(key, "array of " + MetadataTypeName(array->type), "array of integers");
    if (array->has_negative)
        FailField(key, "negative array element", "non-negative integers");
    return array->values;
}

}  // namespace flm::rai
