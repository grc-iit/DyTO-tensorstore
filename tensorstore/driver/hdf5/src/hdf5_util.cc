#include "tensorstore/driver/hdf5/hdf5_util.h"

#include "absl/status/status.h"
#include "tensorstore/util/str_cat.h"

namespace tensorstore {
namespace hdf5_driver {

Result<hid_t> OpenHDF5File(const std::string& path, unsigned flags) {
    hid_t file_id = H5Fopen(path.c_str(), flags, H5P_DEFAULT);
    if (file_id < 0) {
        return absl::InternalError(
            tensorstore::StrCat("Failed to open HDF5 file: ", path));
    }
    return file_id;
}

void CloseHDF5File(hid_t file_id) {
    if (file_id >= 0) {
        H5Fclose(file_id);
    }
}

Result<hid_t> OpenDataset(hid_t file_id, const std::string& name) {
    if (file_id < 0) {
        return absl::InvalidArgumentError("Invalid file ID");
    }
    
    hid_t dataset_id = H5Dopen2(file_id, name.c_str(), H5P_DEFAULT);
    if (dataset_id < 0) {
        return absl::InternalError(
            tensorstore::StrCat("Failed to open dataset: ", name));
    }
    return dataset_id;
}

void CloseDataset(hid_t dataset_id) {
    if (dataset_id >= 0) {
        H5Dclose(dataset_id);
    }
}

}  // namespace hdf5_driver
}  // namespace tensorstore
