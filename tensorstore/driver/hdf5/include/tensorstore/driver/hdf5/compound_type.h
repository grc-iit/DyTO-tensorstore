#ifndef TENSORSTORE_DRIVER_HDF5_COMPOUND_TYPE_H_
#define TENSORSTORE_DRIVER_HDF5_COMPOUND_TYPE_H_

#include <string>
#include <vector>

#include "tensorstore/data_type.h"
#include "tensorstore/util/result.h"
#include "hdf5.h"

namespace tensorstore {
namespace hdf5_driver {

/// Represents a field in a compound data type
struct CompoundTypeField {
    std::string name;     ///< Field name
    size_t offset;        ///< Byte offset in compound type
    DataType dtype;       ///< Field data type
};

/// Manages compound data types
class CompoundType {
public:
    /// Add a field to the compound type
    /// @param field Field to add
    void AddField(const CompoundTypeField& field) {
        fields_.push_back(field);
    }
    
    /// Create HDF5 compound type
    /// @return HDF5 type identifier
    Result<hid_t> CreateHDF5Type() const;
    
    /// Get total size of compound type
    size_t GetTotalSize() const;
    
    /// Get fields
    const std::vector<CompoundTypeField>& fields() const { return fields_; }

private:
    std::vector<CompoundTypeField> fields_;
};

/// Create a variable-length type
/// @param base_type Base data type
/// @return HDF5 type identifier
Result<hid_t> CreateVLenType(DataType base_type);

}  // namespace hdf5_driver
}  // namespace tensorstore

#endif  // TENSORSTORE_DRIVER_HDF5_COMPOUND_TYPE_H_
