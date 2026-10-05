#include "rai/weight_cache.hpp"

#include <nlohmann/json.hpp>

#include <cstdlib>
#include <fstream>
#include <string_view>

namespace flm::rai {
namespace {

/// bumped when the on-disk layout changes in a way older indexes cannot express
constexpr int kIndexFormat = 1;

bool IsDisabled(std::string_view value) {
    return value == "0" || value == "off" || value == "OFF" || value == "false";
}

}  // namespace

WeightCacheKey MakeWeightCacheKey(const std::filesystem::path& gguf_path,
                                  std::uint32_t corelib_major,
                                  std::uint32_t corelib_minor,
                                  std::uint32_t corelib_patch,
                                  std::string layout,
                                  std::uint64_t weight_count) {
    WeightCacheKey key;
    key.corelib_major = corelib_major;
    key.corelib_minor = corelib_minor;
    key.corelib_patch = corelib_patch;
    key.layout = std::move(layout);
    key.weight_count = weight_count;
    std::error_code error;
    const auto size = std::filesystem::file_size(gguf_path, error);
    if (!error) key.gguf_size = size;
    const auto written = std::filesystem::last_write_time(gguf_path, error);
    if (!error) key.gguf_write_time = written.time_since_epoch().count();
    return key;
}

std::optional<WeightCache> WeightCache::ForGguf(const std::filesystem::path& gguf_path) {
    const auto stem = gguf_path.stem().string() + ".rai-weights";
    const char* configured = std::getenv("FLM_RAI_WEIGHT_CACHE");
    if (configured && *configured) {
        if (IsDisabled(configured)) return std::nullopt;
        return WeightCache(configured, stem);
    }
    return WeightCache(gguf_path.parent_path(), stem);
}

WeightCache::WeightCache(std::filesystem::path directory, std::string stem)
    : directory_(std::move(directory)), stem_(std::move(stem)) {}

std::filesystem::path WeightCache::DataPath() const { return directory_ / (stem_ + ".bin"); }

std::filesystem::path WeightCache::IndexPath() const { return directory_ / (stem_ + ".json"); }

std::uint64_t WeightCache::Remove() const {
    std::uint64_t reclaimed = 0;
    const auto data = DataPath().string();
    const auto index = IndexPath().string();
    for (const auto& name : {data, data + ".tmp", index, index + ".tmp"}) {
        std::error_code error;
        const auto size = std::filesystem::file_size(name, error);
        if (error) continue;
        if (std::filesystem::remove(name, error) && !error) reclaimed += size;
    }
    return reclaimed;
}

std::optional<std::vector<CachedWeightSpan>> WeightCache::ReadIndex(
    const WeightCacheKey& expected) const {
    try {
        std::ifstream input(IndexPath(), std::ios::binary);
        if (!input) return std::nullopt;
        const auto document = nlohmann::json::parse(input, nullptr, false);
        if (document.is_discarded() || document.value("format", 0) != kIndexFormat)
            return std::nullopt;

        WeightCacheKey key;
        key.gguf_size = document.value("gguf_size", std::uint64_t{0});
        key.gguf_write_time = document.value("gguf_write_time", std::int64_t{0});
        key.corelib_major = document.value("corelib_major", std::uint32_t{0});
        key.corelib_minor = document.value("corelib_minor", std::uint32_t{0});
        key.corelib_patch = document.value("corelib_patch", std::uint32_t{0});
        key.layout = document.value("layout", std::string());
        key.weight_count = document.value("weight_count", std::uint64_t{0});
        if (!(key == expected)) return std::nullopt;

        const auto spans = document.find("spans");
        if (spans == document.end() || !spans->is_array() || spans->size() != expected.weight_count)
            return std::nullopt;
        std::error_code error;
        const auto data_size = std::filesystem::file_size(DataPath(), error);
        if (error) return std::nullopt;
        std::vector<CachedWeightSpan> result;
        result.reserve(spans->size());
        for (const auto& span : *spans) {
            if (!span.is_object()) return std::nullopt;
            CachedWeightSpan entry{span.value("offset", std::uint64_t{0}),
                                   span.value("size", std::uint64_t{0})};
            // A span past the end means the index describes a file that was
            // truncated under it.
            if (entry.size == 0 || entry.offset + entry.size > data_size) return std::nullopt;
            result.push_back(entry);
        }
        return result;
    } catch (...) {
        return std::nullopt;
    }
}

bool WeightCache::WriteIndex(const WeightCacheKey& key,
                             const std::vector<CachedWeightSpan>& spans) const {
    try {
        nlohmann::json document;
        document["format"] = kIndexFormat;
        document["gguf_size"] = key.gguf_size;
        document["gguf_write_time"] = key.gguf_write_time;
        document["corelib_major"] = key.corelib_major;
        document["corelib_minor"] = key.corelib_minor;
        document["corelib_patch"] = key.corelib_patch;
        document["layout"] = key.layout;
        document["weight_count"] = key.weight_count;
        auto array = nlohmann::json::array();
        for (const auto& span : spans) array.push_back({{"offset", span.offset}, {"size", span.size}});
        document["spans"] = std::move(array);

        const auto final_path = IndexPath();
        const auto temporary = final_path.string() + ".tmp";
        {
            std::ofstream output(temporary, std::ios::binary | std::ios::trunc);
            if (!output) return false;
            output << document.dump();
            if (!output) return false;
        }
        std::error_code error;
        std::filesystem::rename(temporary, final_path, error);
        if (error) {
            std::filesystem::remove(temporary, error);
            return false;
        }
        return true;
    } catch (...) {
        return false;
    }
}

}  // namespace flm::rai
