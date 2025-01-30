#include "tensorstore/driver/hdf5/schema.h"

#include "tensorstore/util/str_cat.h"

namespace tensorstore {
namespace hdf5_driver {

Result<Schema> GetSchemaFromHDF5(const HDF5Metadata& metadata) {
    Schema schema;
    
    // Set data type
    TENSORSTORE_ASSIGN_OR_RETURN(auto dtype, ConvertHDF5Type(metadata.h5_type));
    schema.Set(dtype);
    
    // Set rank and dimensions
    schema.Set(RankConstraint(metadata.rank));
    auto dims = schema.Set(DimensionConstraints());
    
    for (DimensionIndex i = 0; i < metadata.rank; ++i) {
        dims[i].Set(DimensionConstraints::Bounds(0, metadata.shape[i]));
        if (!metadata.dimension_labels[i].empty()) {
            dims[i].Set(DimensionConstraints::Label(metadata.dimension_labels[i]));
        }
    }
    
    // Set chunk layout if chunking is enabled
    if (!metadata.chunks.empty()) {
        auto chunk_layout = schema.Set(ChunkLayout());
        chunk_layout.Set(ChunkLayout::GridOrigin(std::vector<Index>(metadata.rank, 0)));
        chunk_layout.Set(ChunkLayout::ChunkShape(metadata.chunks));
    }
    
    return schema;
}

Result<void> ValidateSchema(const Schema& schema) {
    // Validate rank
    if (!schema.rank().valid()) {
        return absl::InvalidArgumentError("Schema must specify rank");
    }
    
    // Validate data type
    if (!schema.dtype().valid()) {
        return absl::InvalidArgumentError("Schema must specify data type");
    }
    
    // Validate rank constraints
    if (schema.rank().value() < 1) {
        return absl::InvalidArgumentError(
            tensorstore::StrCat("Invalid rank: ", schema.rank().value()));
    }
    
    // Validate dimension properties
    auto dims = schema.dimension_properties();
    if (dims.valid()) {
        for (DimensionIndex i = 0; i < schema.rank().value(); ++i) {
            // Validate dimension size if specified
            if (dims[i].size().valid() && dims[i].size().value() < 0) {
                return absl::InvalidArgumentError(
                    tensorstore::StrCat("Invalid dimension size for dimension ", i));
            }
            
            // Validate chunk size if specified
            if (dims[i].chunk_size().valid() && dims[i].chunk_size().value() <= 0) {
                return absl::InvalidArgumentError(
                    tensorstore::StrCat("Invalid chunk size for dimension ", i));
            }
        }
    }
    
    return absl::OkStatus();
}

Result<DataType> ConvertHDF5Type(hid_t h5_type) {
    H5T_class_t type_class = H5Tget_class(h5_type);
    size_t size = H5Tget_size(h5_type);
    
    switch (type_class) {
        case H5T_INTEGER: {
            H5T_sign_t sign = H5Tget_sign(h5_type);
            if (sign == H5T_SGN_NONE) {  // Unsigned
                switch (size) {
                    case 1: return dtype_v<uint8_t>;
                    case 2: return dtype_v<uint16_t>;
                    case 4: return dtype_v<uint32_t>;
                    case 8: return dtype_v<uint64_t>;
                }
            } else {  // Signed
                switch (size) {
                    case 1: return dtype_v<int8_t>;
                    case 2: return dtype_v<int16_t>;
                    case 4: return dtype_v<int32_t>;
                    case 8: return dtype_v<int64_t>;
                }
            }
            break;
        }
        case H5T_FLOAT: {
            switch (size) {
                case 4: return dtype_v<float>;
                case 8: return dtype_v<double>;
            }
            break;
        }
        case H5T_STRING: {
            // For now, we only support fixed-length strings
            if (H5Tis_variable_str(h5_type)) {
                return absl::InvalidArgumentError("Variable-length strings not supported");
            }
            // TODO: Add string type support
            break;
        }
        case H5T_COMPOUND:
            return absl::InvalidArgumentError("Compound types not supported");
        case H5T_ENUM:
            if (H5Tequal(h5_type, H5T_NATIVE_HBOOL) > 0) {
                return dtype_v<bool>;
            }
            return absl::InvalidArgumentError("Enum types not supported");
        case H5T_ARRAY:
        case H5T_TIME:
        case H5T_BITFIELD:
        case H5T_OPAQUE:
        case H5T_REFERENCE:
        case H5T_VLEN:
            return absl::InvalidArgumentError("Unsupported HDF5 type");
    }
    return absl::InvalidArgumentError(
        tensorstore::StrCat("Unsupported HDF5 type class: ", type_class));
}

Result<hid_t> ConvertToHDF5Type(DataType dtype) {
    if (!dtype.valid()) {
        return absl::InvalidArgumentError("Invalid data type");
    }
    
    hid_t h5_type = -1;
    
    // Map TensorStore types to HDF5 types
    if (dtype == dtype_v<bool>) {
        h5_type = H5T_NATIVE_HBOOL;
    } else if (dtype == dtype_v<int8_t>) {
        h5_type = H5T_NATIVE_INT8;
    } else if (dtype == dtype_v<int16_t>) {
        h5_type = H5T_NATIVE_INT16;
    } else if (dtype == dtype_v<int32_t>) {
        h5_type = H5T_NATIVE_INT32;
    } else if (dtype == dtype_v<int64_t>) {
        h5_type = H5T_NATIVE_INT64;
    } else if (dtype == dtype_v<uint8_t>) {
        h5_type = H5T_NATIVE_UINT8;
    } else if (dtype == dtype_v<uint16_t>) {
        h5_type = H5T_NATIVE_UINT16;
    } else if (dtype == dtype_v<uint32_t>) {
        h5_type = H5T_NATIVE_UINT32;
    } else if (dtype == dtype_v<uint64_t>) {
        h5_type = H5T_NATIVE_UINT64;
    } else if (dtype == dtype_v<float>) {
        h5_type = H5T_NATIVE_FLOAT;
    } else if (dtype == dtype_v<double>) {
        h5_type = H5T_NATIVE_DOUBLE;
    } else {
        return absl::InvalidArgumentError(
            tensorstore::StrCat("Unsupported TensorStore type: ", dtype.name()));
    }
    
    // Create a copy of the type to ensure it persists
    hid_t copied_type = H5Tcopy(h5_type);
    if (copied_type < 0) {
        return absl::InternalError("Failed to copy HDF5 type");
    }
    
    return copied_type;
}

}  // namespace hdf5_driver
}  // namespace tensorstore
