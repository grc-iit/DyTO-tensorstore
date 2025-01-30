#include "tensorstore/driver/hdf5/chunk_cache_v2.h"

#include <numeric>
#include "tensorstore/driver/hdf5/hdf5_util.h"
#include "tensorstore/driver/hdf5/schema.h"
#include "tensorstore/util/str_cat.h"

namespace tensorstore {
namespace hdf5_driver {

namespace {
// Helper function to convert chunk indices to HDF5 offset
std::vector<hsize_t> ConvertIndicesToOffset(span<const Index> indices,
                                          span<const Index> chunk_shape) {
    std::vector<hsize_t> offset(indices.size());
    for (size_t i = 0; i < indices.size(); ++i) {
        offset[i] = static_cast<hsize_t>(indices[i] * chunk_shape[i]);
    }
    return offset;
}
}  // namespace

HDF5ChunkCacheV2::HDF5ChunkCacheV2(hid_t dataset_id, const HDF5Metadata& metadata,
                                 WritePolicy write_policy,
                                 std::chrono::milliseconds write_interval)
    : dataset_id_(dataset_id), metadata_(metadata),
      write_policy_(write_policy), write_interval_(write_interval) {
    // Convert TensorStore type to HDF5 type
    auto h5_type_result = ConvertToHDF5Type(metadata.dtype);
    if (!h5_type_result.ok()) {
        h5_type_ = -1;
        return;
    }
    h5_type_ = *h5_type_result;

    // Initialize the cache with appropriate parameters
    Cache::Limits limits;
    limits.total_bytes_limit = metadata.cache_size_bytes;
    cache_ = std::make_shared<Cache>("hdf5_chunk_cache", limits);
}

HDF5ChunkCacheV2::~HDF5ChunkCacheV2() {
    StopBackgroundWriteback();
    Flush().IgnoreError();
    if (h5_type_ >= 0) {
        H5Tclose(h5_type_);
    }
}

Result<hid_t> HDF5ChunkCacheV2::CreateMemorySpace(span<const hsize_t> count) {
    hid_t memspace = H5Screate_simple(count.size(), count.data(), nullptr);
    if (memspace < 0) {
        return absl::InternalError("Failed to create memory space");
    }
    return memspace;
}

Result<hid_t> HDF5ChunkCacheV2::CreateFileSpace(span<const hsize_t> offset,
                                               span<const hsize_t> count) {
    hid_t filespace = H5Dget_space(dataset_id_);
    if (filespace < 0) {
        return absl::InternalError("Failed to get dataset space");
    }

    herr_t status = H5Sselect_hyperslab(filespace, H5S_SELECT_SET,
                                       offset.data(), nullptr,
                                       count.data(), nullptr);
    if (status < 0) {
        H5Sclose(filespace);
        return absl::InternalError("Failed to select hyperslab");
    }

    return filespace;
}

Result<std::pair<std::vector<hsize_t>, std::vector<hsize_t>>>
HDF5ChunkCacheV2::CalculateChunkCoordinates(const ChunkKey& key) {
    std::vector<hsize_t> offset(metadata_.rank);
    std::vector<hsize_t> count(metadata_.rank);
    
    for (size_t i = 0; i < metadata_.rank; ++i) {
        offset[i] = key.indices[i] * metadata_.chunk_shape[i];
        count[i] = std::min(metadata_.chunk_shape[i],
                           metadata_.shape[i] - offset[i]);
    }
    
    return std::make_pair(offset, count);
}

CacheEntry* HDF5ChunkCacheV2::GetCacheEntry(const ChunkKey& key) {
    auto cache_key = tensorstore::StrCat("chunk:", key.indices[0]);
    for (size_t i = 1; i < key.indices.size(); ++i) {
        tensorstore::StrAppend(&cache_key, ",", key.indices[i]);
    }
    return static_cast<CacheEntry*>(cache_->GetOrCreateEntry(cache_key).get());
}

Future<std::vector<unsigned char>> HDF5ChunkCacheV2::ReadChunkFromHDF5(
    const ChunkKey& key) {
    TENSORSTORE_ASSIGN_OR_RETURN(auto coords, CalculateChunkCoordinates(key));
    auto& [offset, count] = coords;

    // Allocate space for chunk data
    size_t chunk_size = std::accumulate(count.begin(), count.end(),
                                      metadata_.dtype.size(),
                                      std::multiplies<size_t>());
    std::vector<unsigned char> data(chunk_size);

    // Create memory and file spaces
    TENSORSTORE_ASSIGN_OR_RETURN(auto memspace, CreateMemorySpace(count));
    TENSORSTORE_ASSIGN_OR_RETURN(auto filespace, CreateFileSpace(offset, count));

    // Read the data
    herr_t status = H5Dread(dataset_id_, h5_type_, memspace, filespace,
                           H5P_DEFAULT, data.data());

    // Clean up
    H5Sclose(memspace);
    H5Sclose(filespace);

    if (status < 0) {
        return MakeReadyFuture<std::vector<unsigned char>>(
            absl::InternalError("Failed to read chunk from HDF5"));
    }

    return MakeReadyFuture<std::vector<unsigned char>>(std::move(data));
}

HDF5ChunkCacheV2::ReadFuture HDF5ChunkCacheV2::ReadChunk(ChunkKey key) {
    auto entry = GetCacheEntry(key);
    
    if (!entry->chunk_data.data.empty()) {
        // Cache hit
        return MakeReadyFuture<ChunkData>(entry->chunk_data);
    }

    // Cache miss - read from HDF5
    return ReadChunkFromHDF5(key).Then([this, entry](
        Result<std::vector<unsigned char>> data) -> Result<ChunkData> {
        if (!data.ok()) {
            return data.status();
        }
        
        entry->chunk_data.data = std::move(*data);
        entry->chunk_data.dtype = metadata_.dtype;
        entry->chunk_data.shape = std::vector<Index>(
            metadata_.chunk_shape.begin(), metadata_.chunk_shape.end());
        
        return entry->chunk_data;
    });
}

Future<void> HDF5ChunkCacheV2::WriteChunkToHDF5(
    const ChunkKey& key, const std::vector<unsigned char>& data) {
    TENSORSTORE_ASSIGN_OR_RETURN(auto coords, CalculateChunkCoordinates(key));
    auto& [offset, count] = coords;

    // Create memory and file spaces
    TENSORSTORE_ASSIGN_OR_RETURN(auto memspace, CreateMemorySpace(count));
    TENSORSTORE_ASSIGN_OR_RETURN(auto filespace, CreateFileSpace(offset, count));

    // Write the data
    herr_t status = H5Dwrite(dataset_id_, h5_type_, memspace, filespace,
                            H5P_DEFAULT, data.data());

    // Clean up
    H5Sclose(memspace);
    H5Sclose(filespace);

    if (status < 0) {
        return MakeReadyFuture<void>(
            absl::InternalError("Failed to write chunk to HDF5"));
    }

    return MakeReadyFuture<void>(absl::OkStatus());
}

HDF5ChunkCacheV2::WriteFuture HDF5ChunkCacheV2::WriteChunk(
    ChunkKey key, ChunkData data) {
    auto entry = GetCacheEntry(key);
    entry->chunk_data = std::move(data);
    entry->dirty = true;

    if (write_policy_ == WritePolicy::WriteThrough) {
        return WriteChunkToHDF5(key, entry->chunk_data.data);
    }
    return MakeReadyFuture<void>(absl::OkStatus());
}

std::vector<HDF5ChunkCacheV2::CacheEntry*> HDF5ChunkCacheV2::CollectDirtyEntries() {
    std::vector<CacheEntry*> dirty_entries;
    cache_->Visit([&](const Cache::EntryData& entry_data) {
        auto entry = static_cast<CacheEntry*>(entry_data.entry.get());
        if (entry->dirty) {
            dirty_entries.push_back(entry);
        }
    });
    return dirty_entries;
}

void HDF5ChunkCacheV2::StartBackgroundWriteback() {
    if (write_policy_ != WritePolicy::WriteBack) return;
    
    running_ = true;
    writeback_thread_ = std::thread([this]() {
        while (running_) {
            auto dirty_entries = CollectDirtyEntries();
            for (auto entry : dirty_entries) {
                // Extract key from cache key
                std::string cache_key = entry->cache_key();
                std::vector<Index> indices;
                size_t pos = cache_key.find(':');
                std::string indices_str = cache_key.substr(pos + 1);
                
                std::stringstream ss(indices_str);
                std::string index_str;
                while (std::getline(ss, index_str, ',')) {
                    indices.push_back(std::stoi(index_str));
                }
                
                ChunkKey key{indices};
                WriteChunkToHDF5(key, entry->chunk_data.data).IgnoreError();
            }
            std::this_thread::sleep_for(write_interval_);
        }
    });
}

void HDF5ChunkCacheV2::StopBackgroundWriteback() {
    running_ = false;
    if (writeback_thread_.joinable()) {
        writeback_thread_.join();
    }
}

Result<void> HDF5ChunkCacheV2::Flush() {
    auto dirty_entries = CollectDirtyEntries();
    for (auto entry : dirty_entries) {
        // Extract key from cache key (similar to background writeback)
        std::string cache_key = entry->cache_key();
        std::vector<Index> indices;
        size_t pos = cache_key.find(':');
        std::string indices_str = cache_key.substr(pos + 1);
        
        std::stringstream ss(indices_str);
        std::string index_str;
        while (std::getline(ss, index_str, ',')) {
            indices.push_back(std::stoi(index_str));
        }
        
        ChunkKey key{indices};
        TENSORSTORE_RETURN_IF_ERROR(
            WriteChunkToHDF5(key, entry->chunk_data.data).result());
    }
    return absl::OkStatus();
}

}  // namespace hdf5_driver
}  // namespace tensorstore
