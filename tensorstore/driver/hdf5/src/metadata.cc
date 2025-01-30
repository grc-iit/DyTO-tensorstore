#include "tensorstore/driver/hdf5/metadata.h"

#include "tensorstore/driver/hdf5/hdf5_util.h"
#include "tensorstore/util/str_cat.h"

namespace tensorstore {
namespace hdf5_driver {

Result<HDF5Metadata> HDF5Metadata::Open(const std::string& path) {
    HDF5Metadata metadata;
    
    // Open the HDF5 file
    TENSORSTORE_ASSIGN_OR_RETURN(metadata.file_id, OpenHDF5File(path, H5F_ACC_RDONLY));
    
    // Open the root dataset (assuming it's the root group for now)
    TENSORSTORE_ASSIGN_OR_RETURN(metadata.dataset_id, OpenDataset(metadata.file_id, "/"));
    
    // Get the dataspace
    hid_t dataspace = H5Dget_space(metadata.dataset_id);
    if (dataspace < 0) {
        CloseHDF5File(metadata.file_id);
        return absl::InternalError("Failed to get dataspace");
    }
    
    // Get rank and dimensions
    metadata.rank = H5Sget_simple_extent_ndims(dataspace);
    if (metadata.rank < 0) {
        H5Sclose(dataspace);
        CloseHDF5File(metadata.file_id);
        return absl::InternalError("Failed to get rank");
    }
    
    // Get shape
    std::vector<hsize_t> dims(metadata.rank);
    if (H5Sget_simple_extent_dims(dataspace, dims.data(), nullptr) < 0) {
        H5Sclose(dataspace);
        CloseHDF5File(metadata.file_id);
        return absl::InternalError("Failed to get dimensions");
    }
    metadata.shape.resize(metadata.rank);
    for (DimensionIndex i = 0; i < metadata.rank; ++i) {
        metadata.shape[i] = static_cast<Index>(dims[i]);
    }
    
    // Get datatype
    hid_t datatype = H5Dget_type(metadata.dataset_id);
    if (datatype < 0) {
        H5Sclose(dataspace);
        CloseHDF5File(metadata.file_id);
        return absl::InternalError("Failed to get datatype");
    }
    // TODO: Convert HDF5 datatype to tensorstore DataType
    // metadata.dtype = ConvertHDF5Type(datatype);
    H5Tclose(datatype);
    
    // Get chunking information
    hid_t plist = H5Dget_create_plist(metadata.dataset_id);
    if (plist < 0) {
        H5Sclose(dataspace);
        CloseHDF5File(metadata.file_id);
        return absl::InternalError("Failed to get dataset creation property list");
    }
    
    if (H5D_CHUNKED == H5Pget_layout(plist)) {
        std::vector<hsize_t> chunk_dims(metadata.rank);
        if (H5Pget_chunk(plist, metadata.rank, chunk_dims.data()) < 0) {
            H5Pclose(plist);
            H5Sclose(dataspace);
            CloseHDF5File(metadata.file_id);
            return absl::InternalError("Failed to get chunk dimensions");
        }
        metadata.chunks.resize(metadata.rank);
        for (DimensionIndex i = 0; i < metadata.rank; ++i) {
            metadata.chunks[i] = static_cast<Index>(chunk_dims[i]);
        }
    }
    
    H5Pclose(plist);
    H5Sclose(dataspace);
    
    // TODO: Read dimension labels if available
    
    return metadata;
}

Result<void> HDF5Metadata::Close() {
    if (dataset_id >= 0) {
        CloseDataset(dataset_id);
        dataset_id = -1;
    }
    if (file_id >= 0) {
        CloseHDF5File(file_id);
        file_id = -1;
    }
    return absl::OkStatus();
}

}  // namespace hdf5_driver
}  // namespace tensorstore
