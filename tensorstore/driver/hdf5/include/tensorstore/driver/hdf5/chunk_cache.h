#ifndef TENSORSTORE_DRIVER_HDF5_CHUNK_CACHE_H_
#define TENSORSTORE_DRIVER_HDF5_CHUNK_CACHE_H_

#include <memory>
#include <vector>
#include <chrono>

#include "absl/status/status.h"
#include "tensorstore/data_type.h"
#include "tensorstore/driver/hdf5/metadata.h"
#include "tensorstore/index.h"
#include "tensorstore/internal/cache/cache.h"
#include "tensorstore/internal/chunk_cache.h"
#include "tensorstore/util/result.h"
#include "tensorstore/util/span.h"
#include "tensorstore/util/future.h"

namespace tensorstore {
namespace hdf5_driver {

/// Type alias for chunk keys
using ChunkKey = span<const Index>;
/// Type alias for read futures
using ReadFuture = Future<std::vector<unsigned char>>;

/// Cache implementation for HDF5 chunks
class HDF5ChunkCache {
public:
    explicit HDF5ChunkCache(hid_t dataset_id, const HDF5Metadata& metadata);
    ~HDF5ChunkCache();

    /// Reads a chunk from the HDF5 dataset
    /// @param chunk_indices The indices of the chunk to read
    /// @return Result containing the chunk data or an error
    Result<std::vector<unsigned char>> ReadChunk(span<const Index> chunk_indices);

    /// Writes a chunk to the HDF5 dataset
    /// @param chunk_indices The indices of the chunk to write
    /// @param data The data to write
    /// @return Status indicating success or failure
    Result<void> WriteChunk(span<const Index> chunk_indices, 
                           span<const unsigned char> data);

    /// Reads multiple chunks in parallel
    /// @param keys Vector of chunk indices to read
    /// @return Future that resolves when all chunks are read
    ReadFuture ReadMultipleChunks(span<const ChunkKey> keys);

    /// Prefetches chunks into cache asynchronously
    /// @param keys Vector of chunk indices to prefetch
    void Prefetch(span<const ChunkKey> keys);

    /// Statistics about the cache state
    struct CacheStats {
        size_t num_entries{0};    ///< Number of entries in cache
        size_t total_size{0};     ///< Total size of cached data in bytes
        size_t num_dirty{0};      ///< Number of dirty entries
        size_t num_hits{0};       ///< Number of cache hits
        size_t num_misses{0};     ///< Number of cache misses
    };

    /// Get current cache statistics
    /// @return Current cache statistics
    CacheStats GetStats() const;

    /// Evict entries from cache until total size is below target_size
    /// @param target_size Target cache size in bytes
    void EvictEntries(size_t target_size);

private:
    /// Creates a memory space for chunk transfer
    /// @param count The dimensions of the chunk
    /// @return Result containing the memory space ID or an error
    Result<hid_t> CreateMemorySpace(span<const hsize_t> count);

    /// Creates a file space for chunk transfer
    /// @param offset The chunk offset in the dataset
    /// @param count The dimensions of the chunk
    /// @return Result containing the file space ID or an error
    Result<hid_t> CreateFileSpace(span<const hsize_t> offset, 
                                 span<const hsize_t> count);

    /// Calculates the size of a chunk in bytes
    size_t GetChunkSizeInBytes() const;

    /// Gets or creates a cache entry for the given key
    /// @param key Chunk indices
    /// @return Pointer to cache entry
    CacheEntry* GetCacheEntry(const ChunkKey& key);

    /// Internal cache entry structure
    struct CacheEntry {
        std::vector<unsigned char> data;
        bool dirty{false};
        std::chrono::steady_clock::time_point last_access;
        
        size_t ComputeSize() const { return data.size(); }
        void DoWrite();
    };

    /// The actual cache storage
    std::unique_ptr<internal::Cache<CacheEntry>> cache_;

    hid_t dataset_id_;  ///< HDF5 dataset identifier
    HDF5Metadata metadata_;  ///< Dataset metadata
    hid_t h5_type_;  ///< HDF5 datatype identifier
};

}  // namespace hdf5_driver
}  // namespace tensorstore

#endif  // TENSORSTORE_DRIVER_HDF5_CHUNK_CACHE_H_
