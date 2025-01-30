#ifndef TENSORSTORE_DRIVER_HDF5_HDF5_UTIL_H_
#define TENSORSTORE_DRIVER_HDF5_HDF5_UTIL_H_

#include <string>

#include "hdf5.h"
#include "tensorstore/util/result.h"

namespace tensorstore {
namespace hdf5_driver {

/// Opens an HDF5 file with the specified flags.
/// @param path Path to the HDF5 file.
/// @param flags Access flags for opening the file.
/// @return Result containing the file ID or an error.
Result<hid_t> OpenHDF5File(const std::string& path, unsigned flags);

/// Safely closes an HDF5 file.
/// @param file_id The HDF5 file ID to close.
void CloseHDF5File(hid_t file_id);

/// Opens a dataset within an HDF5 file.
/// @param file_id The HDF5 file ID.
/// @param name Name of the dataset to open.
/// @return Result containing the dataset ID or an error.
Result<hid_t> OpenDataset(hid_t file_id, const std::string& name);

/// Safely closes an HDF5 dataset.
/// @param dataset_id The HDF5 dataset ID to close.
void CloseDataset(hid_t dataset_id);

}  // namespace hdf5_driver
}  // namespace tensorstore

#endif  // TENSORSTORE_DRIVER_HDF5_HDF5_UTIL_H_
