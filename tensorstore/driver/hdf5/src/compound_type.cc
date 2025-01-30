#include "tensorstore/driver/hdf5/compound_type.h"

#include "tensorstore/driver/hdf5/schema.h"
#include "tensorstore/util/str_cat.h"

namespace tensorstore {
namespace hdf5_driver {

size_t CompoundType::GetTotalSize() const {
    size_t total_size = 0;
    for (const auto& field : fields_) {
        total_size += field.dtype.size();
    }
    return total_size;
}

Result<hid_t> CompoundType::CreateHDF5Type() const {
    size_t total_size = GetTotalSize();
    
    // Create compound datatype
    hid_t type_id = H5Tcreate(H5T_COMPOUND, total_size);
    if (type_id < 0) {
        return absl::InternalError("Failed to create compound type");
    }
    
    // Add fields
    for (const auto& field : fields_) {
        TENSORSTORE_ASSIGN_OR_RETURN(
            auto field_type_id,
            ConvertToHDF5Type(field.dtype));
        
        herr_t status = H5Tinsert(type_id, field.name.c_str(),
                                field.offset, field_type_id);
        
        if (status < 0) {
            H5Tclose(type_id);
            return absl::InternalError(
                tensorstore::StrCat("Failed to insert field: ", field.name));
        }
        
        H5Tclose(field_type_id);
    }
    
    return type_id;
}

Result<hid_t> CreateVLenType(DataType base_type) {
    // Convert base type to HDF5
    TENSORSTORE_ASSIGN_OR_RETURN(
        auto base_type_id,
        ConvertToHDF5Type(base_type));
    
    // Create variable-length type
    hid_t vlen_type_id = H5Tvlen_create(base_type_id);
    H5Tclose(base_type_id);
    
    if (vlen_type_id < 0) {
        return absl::InternalError(
            tensorstore::StrCat("Failed to create variable-length type for: ",
                              base_type.name()));
    }
    
    return vlen_type_id;
}

}  // namespace hdf5_driver
}  // namespace tensorstore
