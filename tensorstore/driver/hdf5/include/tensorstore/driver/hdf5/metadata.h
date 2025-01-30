#ifndef TENSORSTORE_DRIVER_HDF5_METADATA_H_
#define TENSORSTORE_DRIVER_HDF5_METADATA_H_

#include <string>
#include <vector>

#include "hdf5.h"
#include "tensorstore/data_type.h"
#include "tensorstore/index.h"
#include "tensorstore/rank.h"
#include "tensorstore/util/result.h"

namespace tensorstore {
namespace hdf5_driver {

/// Represents the metadata for an HDF5 dataset
struct HDF5Metadata {
    DimensionIndex rank;
    std::vector<Index> shape;
    std::vector<Index> chunks;
    DataType dtype;
    std::vector<std::string> dimension_labels;
    
    // HDF5 specific handles
    hid_t file_id;
    hid_t dataset_id;
    
    /// Opens an HDF5 file and reads its metadata
    /// @param path Path to the HDF5 file
    /// @return Result containing the metadata or an error
    static Result<HDF5Metadata> Open(const std::string& path);
    
    /// Closes the HDF5 file and dataset handles
    /// @return Result indicating success or failure
    Result<void> Close();
};

}  // namespace hdf5_driver
}  // namespace tensorstore

#endif  // TENSORSTORE_DRIVER_HDF5_METADATA_H_
