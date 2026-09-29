/// \file weight_cache.hpp
/// \brief On-disk cache of a model's packed corelib weights
/// \note Requantizing every weight from Q8_0 is effectively the whole of a rai
///       model load. corelib hands the packed bytes back
///       (ryzenai_corelib_weights_copy_data) and maps them in again later
///       (..._weights_create_from_file), so the refit is paid once per GGUF.
///       One data file plus a JSON index written last and renamed into place:
///       a data file without a matching index is never used.
#pragma once

#include <cstdint>
#include <filesystem>
#include <optional>
#include <string>
#include <vector>

namespace flm::rai {

/// \brief where one packed weight lives inside the cache file
struct CachedWeightSpan {
    std::uint64_t offset{};
    std::uint64_t size{};
};

/// \brief the identity a cache is only valid for
/// \note The GGUF is identified by size and write time, not content: hashing a
///       multi-gigabyte file costs more than the packing the cache avoids.
///       corelib rejects a slice whose length is not exactly what its
///       descriptor packs to, so a stale-but-plausible cache is caught there.
///       `layout` names everything else that shapes the packed bytes -- every
///       group size, every width -- so changing any of them is a miss.
struct WeightCacheKey {
    std::uint64_t gguf_size{};
    std::int64_t gguf_write_time{};
    std::uint32_t corelib_major{}, corelib_minor{}, corelib_patch{};
    std::string layout;
    std::uint64_t weight_count{};

    bool operator==(const WeightCacheKey& other) const = default;
};

WeightCacheKey MakeWeightCacheKey(const std::filesystem::path& gguf_path,
                                  std::uint32_t corelib_major,
                                  std::uint32_t corelib_minor,
                                  std::uint32_t corelib_patch,
                                  std::string layout,
                                  std::uint64_t weight_count);

class WeightCache final {
public:
    /// \brief the cache for one GGUF, or nullopt when caching is disabled
    /// \param gguf_path the model file; the cache is named after it
    /// \note FLM_RAI_WEIGHT_CACHE unset puts the cache beside the GGUF; a path
    ///       redirects it; "0", "off" or "false" disables it. Named after the
    ///       GGUF, so several models can share one redirected directory.
    static std::optional<WeightCache> ForGguf(const std::filesystem::path& gguf_path);

    WeightCache(std::filesystem::path directory, std::string stem);

    std::filesystem::path DataPath() const;
    std::filesystem::path IndexPath() const;

    /// \brief the spans, when an index is present, current and in bounds
    /// \note Never throws: a damaged cache is a miss, not a failed load.
    std::optional<std::vector<CachedWeightSpan>> ReadIndex(const WeightCacheKey& expected) const;

    /// \brief write the index for a data file already renamed into place
    /// \return false when it could not be written
    bool WriteIndex(const WeightCacheKey& key, const std::vector<CachedWeightSpan>& spans) const;

    /// \brief delete this cache and any temporaries an interrupted write left
    /// \return the bytes reclaimed; never throws
    std::uint64_t Remove() const;

private:
    std::filesystem::path directory_;
    std::string stem_;
};

}  // namespace flm::rai
