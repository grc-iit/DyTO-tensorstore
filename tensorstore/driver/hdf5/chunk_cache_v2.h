#ifndef TENSORSTORE_DRIVER_HDF5_CHUNK_CACHE_V2_H_
#define TENSORSTORE_DRIVER_HDF5_CHUNK_CACHE_V2_H_

#include <memory>
#include <vector>
#include <thread>
#include <atomic>
#include <chrono>

#include "absl/container/flat_hash_map.h"
#include "absl/hash/hash.h"
#include "tensorstore/data_type.h"
#include "tensorstore/driver/hdf5/metadata.h"
#include "tensorstore/index.h"
#include "tensorstore/internal/cache/cache.h"
#include "tensorstore/internal/chunk_cache.h"
#include "tensorstore/util/future.h"
#include "tensorstore/util/result.h"
#include "tensorstore/util/span.h"

namespace tensorstore {
namespace hdf5_driver {

/// Enhanced cache implementation for HDF5 chunks with future-based operations
class HDF5ChunkCacheV2 {
public:
    struct ChunkKey {
        std::vector<Index> indices;
        
        template <typename H>
        friend H AbslHashValue(H h, const ChunkKey& key) {
            return H::combine(std::move(h), key.indices);
        }

        bool operator==(const ChunkKey& other) const {
            return indices == other.indices;
        }
    };

    struct ChunkData {
        std::vector<unsigned char> data;
        DataType dtype;
        std::vector<Index> shape;
    };
    
    using ReadFuture = Future<ChunkData>;
    using WriteFuture = Future<void>;
    
    enum class WritePolicy {
        WriteThrough,  // Write immediately to HDF5
        WriteBack     // Cache writes and flush periodically
    };

    explicit HDF5ChunkCacheV2(hid_t dataset_id, const HDF5Metadata& metadata,
                           WritePolicy write_policy = WritePolicy::WriteBack,
                           std::chrono::milliseconds write_interval = std::chrono::seconds(5));
    ~HDF5ChunkCacheV2();

    /// Asynchronously reads a chunk from the cache or HDF5 dataset
    /// @param key The chunk key containing indices
    /// @return Future containing the chunk data or an error
    ReadFuture ReadChunk(ChunkKey key);

    /// Asynchronously writes a chunk to the cache and marks it dirty
    /// @param key The chunk key containing indices
    /// @param data The chunk data to write
    /// @return Future indicating completion or error
    WriteFuture WriteChunk(ChunkKey key, ChunkData data);

    /// Starts the background write-back thread
    void StartBackgroundWriteback();

    /// Stops the background write-back thread
    void StopBackgroundWriteback();

    /// Flushes all dirty entries to HDF5
    Result<void> Flush();

private:
    class CacheEntry : public Cache::Entry {
    public:
        ChunkData chunk_data;
        bool dirty = false;
        
        size_t ComputeSize() override {
            return chunk_data.data.size();
        }
    };

    /// Writes a dirty chunk back to HDF5
    /// @param key The chunk key
    /// @param entry The cache entry containing the chunk data
    /// @return Status indicating success or failure
    Result<void> WriteBackToHDF5(const ChunkKey& key, CacheEntry* entry);

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

    /// Collects all dirty entries from the cache
    std::vector<CacheEntry*> CollectDirtyEntries();

    /// Calculate chunk coordinates for HDF5 access
    Result<std::pair<std::vector<hsize_t>, std::vector<hsize_t>>> 
    CalculateChunkCoordinates(const ChunkKey& key);

    /// Gets or creates a cache entry for the given key
    CacheEntry* GetCacheEntry(const ChunkKey& key);

    /// Reads a chunk directly from HDF5
    Future<std::vector<unsigned char>> ReadChunkFromHDF5(const ChunkKey& key);

    /// Writes a chunk directly to HDF5
    Future<void> WriteChunkToHDF5(const ChunkKey& key, 
                                 const std::vector<unsigned char>& data);

    hid_t dataset_id_;
    hid_t h5_type_;
    HDF5Metadata metadata_;
    std::shared_ptr<Cache> cache_;
    WritePolicy write_policy_;
    std::chrono::milliseconds write_interval_;
    std::atomic<bool> running_{false};
    std::thread writeback_thread_;
};

}  // namespace hdf5_driver
}  // namespace tensorstore

#endif  // TENSORSTORE_DRIVER_HDF5_CHUNK_CACHE_V2_H_
