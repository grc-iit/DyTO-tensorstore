#ifndef TENSORSTORE_DRIVER_HDF5_SCHEMA_H_
#define TENSORSTORE_DRIVER_HDF5_SCHEMA_H_

#include "tensorstore/schema.h"
#include "tensorstore/driver/hdf5/metadata.h"
#include "tensorstore/util/result.h"

namespace tensorstore {
namespace hdf5_driver {

/// Converts HDF5 metadata to tensorstore Schema
/// @param metadata The HDF5 metadata to convert
/// @return Result containing the Schema or an error
Result<Schema> GetSchemaFromHDF5(const HDF5Metadata& metadata);

/// Validates that a Schema is compatible with HDF5 driver requirements
/// @param schema The Schema to validate
/// @return Result indicating success or an error describing validation failures
Result<void> ValidateSchema(const Schema& schema);

/// Schema specification for HDF5 driver
struct HDF5Schema {
    // HDF5-specific schema options can be added here
    // For example: compression settings, chunk cache parameters, etc.
};

/// Converts an HDF5 datatype to a TensorStore DataType
/// @param h5_type HDF5 datatype identifier
/// @return Result containing the TensorStore DataType or an error
Result<DataType> ConvertHDF5Type(hid_t h5_type);

/// Converts a TensorStore DataType to an HDF5 datatype
/// @param dtype TensorStore DataType to convert
/// @return Result containing the HDF5 datatype identifier or an error
Result<hid_t> ConvertToHDF5Type(DataType dtype);

}  // namespace hdf5_driver
}  // namespace tensorstore

#endif  // TENSORSTORE_DRIVER_HDF5_SCHEMA_H_
