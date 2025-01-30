#include "tensorstore/driver/hdf5/driver.h"

#include <memory>
#include "tensorstore/driver/registry.h"
#include "tensorstore/internal/concurrency.h"
#include "tensorstore/util/result.h"
#include "tensorstore/driver/hdf5/chunk_cache.h"

namespace tensorstore {
namespace hdf5_driver {

namespace {
const internal::DriverRegistration<HDF5Driver> registration;
}  // namespace

DataType HDF5Driver::dtype() {
  return metadata_.dtype;
}

DimensionIndex HDF5Driver::rank() {
  return metadata_.rank;
}

Result<Schema> HDF5Driver::GetSchema() {
  return GetSchemaFromHDF5(metadata_);
}

Result<ChunkLayout> HDF5Driver::GetChunkLayout() {
    ChunkLayout layout;
    
    // Set rank
    layout.Set(RankConstraint(metadata_.rank));
    
    // Set chunk dimensions and grid origin
    std::vector<Index> grid_origin(metadata_.rank, 0);
    layout.Set(ChunkLayout::GridOrigin(grid_origin));
    layout.Set(ChunkLayout::ChunkShape(metadata_.chunks));
    
    // Set inner order (default to C-order)
    auto inner_order = layout.Set(ChunkLayout::InnerOrder());
    for (DimensionIndex i = 0; i < metadata_.rank; ++i) {
        inner_order[i] = i;
    }
    
    return layout;
}

void HDF5Driver::Read(ReadRequest request, ReadChunkReceiver receiver) {
  // Create read operation context
  auto context = std::make_shared<ReadContext>(std::move(request));
  
  // Get the chunks to read
  auto chunks = context->GetChunks();
  if (!chunks.ok()) {
    receiver.SetError(chunks.status());
    return;
  }
  
  // Create chunk cache if not exists
  if (!cache_) {
    cache_ = std::make_unique<HDF5ChunkCache>(dataset_id_, metadata_);
  }
  
  // Schedule read operations for each chunk
  for (const auto& chunk : *chunks) {
    // Start an asynchronous read operation
    auto future = internal::PromiseFuturePair<std::vector<unsigned char>>::Make();
    
    // Read the chunk
    auto read_result = cache_->ReadChunk(chunk.indices);
    if (!read_result.ok()) {
      receiver.SetError(read_result.status());
      return;
    }
    
    // Create chunk data
    auto chunk_data = std::make_shared<ChunkData>();
    chunk_data->data = std::move(*read_result);
    
    // Issue the chunk to the receiver
    receiver.IssueChunk(chunk, std::move(chunk_data));
  }
  
  // Mark read operation as complete
  receiver.SetDone();
}

void HDF5Driver::Write(WriteRequest request, WriteChunkReceiver receiver) {
  // Create write operation context
  auto context = std::make_shared<WriteContext>(std::move(request));
  
  // Get the chunks to write
  auto chunks = context->GetChunks();
  if (!chunks.ok()) {
    receiver.SetError(chunks.status());
    return;
  }
  
  // Create chunk cache if not exists
  if (!cache_) {
    cache_ = std::make_unique<HDF5ChunkCache>(dataset_id_, metadata_);
  }
  
  // Schedule write operations for each chunk
  for (const auto& chunk : *chunks) {
    // Write the chunk
    auto write_result = cache_->WriteChunk(chunk.indices, 
                                         span<const unsigned char>(chunk.data->data));
    if (!write_result.ok()) {
      receiver.SetError(write_result.status());
      return;
    }
  }
  
  // Mark write operation as complete
  receiver.SetDone();
}

Result<void> HDF5Driver::WriteMetadata(const std::string& key,
                                     const nlohmann::json& value) {
    AttributeManager attrs(metadata_.dataset_id);
    return attrs.WriteJsonAttribute(key, value);
}

Result<nlohmann::json> HDF5Driver::ReadMetadata(const std::string& key) {
    AttributeManager attrs(metadata_.dataset_id);
    return attrs.ReadJsonAttribute(key);
}

std::vector<std::string> HDF5Driver::ListMetadata() const {
    AttributeManager attrs(metadata_.dataset_id);
    return attrs.ListAttributes();
}

std::string HDF5Driver::GetGroupPath(const std::string& path) {
    size_t pos = path.find_last_of('/');
    if (pos == std::string::npos || pos == 0) {
        return "/";
    }
    return path.substr(0, pos);
}

std::string HDF5Driver::GetBaseName(const std::string& path) {
    size_t pos = path.find_last_of('/');
    if (pos == std::string::npos) {
        return path;
    }
    return path.substr(pos + 1);
}

Result<std::shared_ptr<HDF5Driver>> HDF5Driver::OpenDatasetInGroup(
    const HDF5Group& group, const std::string& name) {
    // Open dataset
    hid_t dataset_id = H5Dopen2(group.id(), name.c_str(), H5P_DEFAULT);
    if (dataset_id < 0) {
        return absl::NotFoundError(
            tensorstore::StrCat("Dataset not found: ", name));
    }
    
    // Get dataset metadata
    HDF5Metadata metadata;
    metadata.dataset_id = dataset_id;
    
    // Get dataspace
    hid_t space_id = H5Dget_space(dataset_id);
    metadata.rank = H5Sget_simple_extent_ndims(space_id);
    
    std::vector<hsize_t> dims(metadata.rank);
    H5Sget_simple_extent_dims(space_id, dims.data(), nullptr);
    metadata.shape.assign(dims.begin(), dims.end());
    
    H5Sclose(space_id);
    
    // Get datatype
    metadata.h5_type = H5Dget_type(dataset_id);
    
    // Get chunk dimensions if chunked
    hid_t plist = H5Dget_create_plist(dataset_id);
    if (H5D_CHUNKED == H5Pget_layout(plist)) {
        std::vector<hsize_t> chunks(metadata.rank);
        H5Pget_chunk(plist, metadata.rank, chunks.data());
        metadata.chunks.assign(chunks.begin(), chunks.end());
    }
    H5Pclose(plist);
    
    return std::make_shared<HDF5Driver>(dataset_id, std::move(metadata));
}

Result<std::shared_ptr<HDF5Driver>> HDF5Driver::OpenDataset(
    hid_t file_id, const std::string& path) {
    std::string group_path = GetGroupPath(path);
    std::string dataset_name = GetBaseName(path);
    
    TENSORSTORE_ASSIGN_OR_RETURN(
        auto group, HDF5Group::Open(file_id, group_path));
    
    TENSORSTORE_ASSIGN_OR_RETURN(
        bool exists, group.HasChild(dataset_name));
    if (!exists) {
        return absl::NotFoundError(
            tensorstore::StrCat("Dataset not found: ", path));
    }
    
    return OpenDatasetInGroup(group, dataset_name);
}

Result<std::shared_ptr<HDF5Driver>> HDF5Driver::CreateDataset(
    hid_t file_id, const std::string& path,
    const HDF5Metadata& metadata) {
    std::string group_path = GetGroupPath(path);
    std::string dataset_name = GetBaseName(path);
    
    // Create or open group
    TENSORSTORE_ASSIGN_OR_RETURN(
        auto group, HDF5Group::Create(file_id, group_path));
    
    // Check if dataset already exists
    TENSORSTORE_ASSIGN_OR_RETURN(
        bool exists, group.HasChild(dataset_name));
    if (exists) {
        return absl::AlreadyExistsError(
            tensorstore::StrCat("Dataset already exists: ", path));
    }
    
    // Create dataspace
    std::vector<hsize_t> dims(metadata.shape.begin(), metadata.shape.end());
    hid_t space_id = H5Screate_simple(metadata.rank, dims.data(), nullptr);
    if (space_id < 0) {
        return absl::InternalError("Failed to create dataspace");
    }
    
    // Set creation properties
    hid_t dcpl = H5Pcreate(H5P_DATASET_CREATE);
    if (!metadata.chunks.empty()) {
        std::vector<hsize_t> chunks(metadata.chunks.begin(), metadata.chunks.end());
        H5Pset_chunk(dcpl, metadata.rank, chunks.data());
    }
    
    // Create dataset
    hid_t dataset_id = H5Dcreate2(group.id(), dataset_name.c_str(),
                                 metadata.h5_type, space_id,
                                 H5P_DEFAULT, dcpl, H5P_DEFAULT);
    
    H5Pclose(dcpl);
    H5Sclose(space_id);
    
    if (dataset_id < 0) {
        return absl::InternalError(
            tensorstore::StrCat("Failed to create dataset: ", path));
    }
    
    return std::make_shared<HDF5Driver>(dataset_id, metadata);
}

}  // namespace hdf5_driver
}  // namespace tensorstore
