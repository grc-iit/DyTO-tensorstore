#include "tensorstore/driver/hdf5/chunk_cache.h"

#include <numeric>
#include "tensorstore/driver/hdf5/hdf5_util.h"
#include "tensorstore/driver/hdf5/schema.h"
#include "tensorstore/util/str_cat.h"

namespace tensorstore {
namespace hdf5_driver {

HDF5ChunkCache::HDF5ChunkCache(hid_t dataset_id, const HDF5Metadata& metadata)
    : dataset_id_(dataset_id), metadata_(metadata) {
    // Convert TensorStore type to HDF5 type
    auto h5_type_result = ConvertToHDF5Type(metadata.dtype);
    if (!h5_type_result.ok()) {
        // Handle error - in production, we might want to throw here
        h5_type_ = -1;
        return;
    }
    h5_type_ = *h5_type_result;
}

HDF5ChunkCache::~HDF5ChunkCache() {
    if (h5_type_ >= 0) {
        H5Tclose(h5_type_);
    }
}

Result<hid_t> HDF5ChunkCache::CreateMemorySpace(span<const hsize_t> count) {
    hid_t memspace = H5Screate_simple(count.size(), count.data(), nullptr);
    if (memspace < 0) {
        return absl::InternalError("Failed to create memory space");
    }
    return memspace;
}

Result<hid_t> HDF5ChunkCache::CreateFileSpace(span<const hsize_t> offset,
                                             span<const hsize_t> count) {
    // Get the dataset space
    hid_t filespace = H5Dget_space(dataset_id_);
    if (filespace < 0) {
        return absl::InternalError("Failed to get dataset space");
    }

    // Select the hyperslab
    herr_t status = H5Sselect_hyperslab(filespace, H5S_SELECT_SET,
                                       offset.data(), nullptr,
                                       count.data(), nullptr);
    if (status < 0) {
        H5Sclose(filespace);
        return absl::InternalError("Failed to select hyperslab");
    }

    return filespace;
}

size_t HDF5ChunkCache::GetChunkSizeInBytes() const {
    return std::accumulate(metadata_.chunks.begin(), metadata_.chunks.end(), 
                          static_cast<size_t>(1), std::multiplies<size_t>()) *
           metadata_.dtype.size();
}

Result<std::vector<unsigned char>> HDF5ChunkCache::ReadChunk(
    span<const Index> chunk_indices) {
    if (h5_type_ < 0) {
        return absl::InternalError("Invalid HDF5 type");
    }

    // Convert chunk indices to HDF5 offsets and counts
    std::vector<hsize_t> offset;
    std::vector<hsize_t> count;
    offset.reserve(chunk_indices.size());
    count.reserve(chunk_indices.size());

    for (size_t i = 0; i < chunk_indices.size(); ++i) {
        offset.push_back(chunk_indices[i] * metadata_.chunks[i]);
        count.push_back(metadata_.chunks[i]);
    }

    // Create memory and file spaces
    TENSORSTORE_ASSIGN_OR_RETURN(auto memspace, CreateMemorySpace(count));
    TENSORSTORE_ASSIGN_OR_RETURN(auto filespace, CreateFileSpace(offset, count));

    // Allocate buffer for chunk data
    std::vector<unsigned char> buffer(GetChunkSizeInBytes());

    // Read the chunk
    herr_t status = H5Dread(dataset_id_, h5_type_, memspace, filespace,
                           H5P_DEFAULT, buffer.data());

    // Clean up
    H5Sclose(memspace);
    H5Sclose(filespace);

    if (status < 0) {
        return absl::InternalError("Failed to read chunk data");
    }

    return buffer;
}

Result<void> HDF5ChunkCache::WriteChunk(span<const Index> chunk_indices,
                                       span<const unsigned char> data) {
    if (h5_type_ < 0) {
        return absl::InternalError("Invalid HDF5 type");
    }

    // Verify data size
    if (data.size() != GetChunkSizeInBytes()) {
        return absl::InvalidArgumentError(
            tensorstore::StrCat("Invalid chunk data size: ", data.size(),
                              " != ", GetChunkSizeInBytes()));
    }

    // Convert chunk indices to HDF5 offsets and counts
    std::vector<hsize_t> offset;
    std::vector<hsize_t> count;
    offset.reserve(chunk_indices.size());
    count.reserve(chunk_indices.size());

    for (size_t i = 0; i < chunk_indices.size(); ++i) {
        offset.push_back(chunk_indices[i] * metadata_.chunks[i]);
        count.push_back(metadata_.chunks[i]);
    }

    // Create memory and file spaces
    TENSORSTORE_ASSIGN_OR_RETURN(auto memspace, CreateMemorySpace(count));
    TENSORSTORE_ASSIGN_OR_RETURN(auto filespace, CreateFileSpace(offset, count));

    // Write the chunk
    herr_t status = H5Dwrite(dataset_id_, h5_type_, memspace, filespace,
                            H5P_DEFAULT, data.data());

    // Clean up
    H5Sclose(memspace);
    H5Sclose(filespace);

    if (status < 0) {
        return absl::InternalError("Failed to write chunk data");
    }

    return absl::OkStatus();
}

void HDF5ChunkCache::CacheEntry::DoWrite() {
    // Implementation will be added in the WriteChunk method
    // This is a placeholder for the eviction process
}

void HDF5ChunkCache::EvictEntries(size_t target_size) {
    std::vector<CacheEntry*> candidates;
    cache_->CollectEvictionCandidates(candidates);
    
    // Sort candidates by last access time (LRU policy)
    std::sort(candidates.begin(), candidates.end(),
              [](const CacheEntry* a, const CacheEntry* b) {
                  return a->last_access < b->last_access;
              });
    
    size_t current_size = cache_->GetSize();
    for (auto* entry : candidates) {
        if (current_size <= target_size) break;
        
        // If entry is dirty, write it back before evicting
        if (entry->dirty) {
            entry->DoWrite();
        }
        
        // Remove entry from cache and update size
        size_t entry_size = entry->ComputeSize();
        cache_->Remove(entry);
        current_size -= entry_size;
    }
}

HDF5ChunkCache::CacheStats HDF5ChunkCache::GetStats() const {
    CacheStats stats;
    
    // Collect statistics from cache entries
    cache_->ForEachEntry([&stats](const CacheEntry* entry) {
        stats.num_entries++;
        stats.total_size += entry->ComputeSize();
        if (entry->dirty) {
            stats.num_dirty++;
        }
    });
    
    // Get hit/miss statistics from cache
    stats.num_hits = cache_->GetHitCount();
    stats.num_misses = cache_->GetMissCount();
    
    return stats;
}

HDF5ChunkCache::CacheEntry* HDF5ChunkCache::GetCacheEntry(const ChunkKey& key) {
    // Create a unique key for the cache entry
    std::string cache_key;
    cache_key.reserve(key.size() * sizeof(Index));
    for (const auto& idx : key) {
        absl::StrAppend(&cache_key, idx, "_");
    }
    
    // Get or create cache entry
    auto* entry = cache_->Get(cache_key);
    if (!entry) {
        entry = cache_->Create(cache_key);
        entry->last_access = std::chrono::steady_clock::now();
    }
    return entry;
}

ReadFuture HDF5ChunkCache::ReadMultipleChunks(span<const ChunkKey> keys) {
    std::vector<ReadFuture> futures;
    futures.reserve(keys.size());
    
    // Create futures for each chunk read operation
    for (const auto& key : keys) {
        auto future = MakeFuture([this, key]() -> Result<std::vector<unsigned char>> {
            return ReadChunk(key);
        });
        futures.push_back(std::move(future));
    }
    
    // Return a future that completes when all reads are done
    return WhenAll(std::move(futures));
}

void HDF5ChunkCache::Prefetch(span<const ChunkKey> keys) {
    for (const auto& key : keys) {
        auto* entry = GetCacheEntry(key);
        
        // Only prefetch if not already in cache
        if (entry->data.empty()) {
            // Start async read operation
            auto future = MakeFuture([this, key]() -> Result<std::vector<unsigned char>> {
                return ReadChunk(key);
            });
            
            // Force immediate execution
            future.Force();
            
            // Update cache entry
            future.ExecuteWhenReady([entry](Result<std::vector<unsigned char>> result) {
                if (result.ok()) {
                    entry->data = std::move(*result);
                    entry->last_access = std::chrono::steady_clock::now();
                }
            });
        }
    }
}

}  // namespace hdf5_driver
}  // namespace tensorstore
