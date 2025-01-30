#include "tensorstore/driver/hdf5/attribute_manager.h"

#include "absl/status/status.h"
#include "tensorstore/serialization/json.h"
#include "tensorstore/util/str_cat.h"

namespace tensorstore {
namespace hdf5_driver {

namespace {

// Helper function to create attribute space
Result<hid_t> CreateAttributeSpace(size_t size) {
    hsize_t dims[] = {size};
    hid_t space_id = H5Screate_simple(1, dims, nullptr);
    if (space_id < 0) {
        return absl::InternalError("Failed to create attribute dataspace");
    }
    return space_id;
}

}  // namespace

Result<void> AttributeManager::WriteAttribute(const std::string& name,
                                           const void* data,
                                           hid_t type_id,
                                           size_t size) {
    // Create or open attribute
    hid_t attr_id;
    if (HasAttribute(name)) {
        attr_id = H5Aopen(object_id_, name.c_str(), H5P_DEFAULT);
    } else {
        TENSORSTORE_ASSIGN_OR_RETURN(auto space_id, CreateAttributeSpace(size));
        attr_id = H5Acreate2(object_id_, name.c_str(), type_id,
                            space_id, H5P_DEFAULT, H5P_DEFAULT);
        H5Sclose(space_id);
    }
    
    if (attr_id < 0) {
        return absl::InternalError(
            tensorstore::StrCat("Failed to create/open attribute: ", name));
    }
    
    // Write data
    herr_t status = H5Awrite(attr_id, type_id, data);
    H5Aclose(attr_id);
    
    if (status < 0) {
        return absl::InternalError(
            tensorstore::StrCat("Failed to write attribute: ", name));
    }
    
    return absl::OkStatus();
}

Result<void> AttributeManager::ReadAttribute(const std::string& name,
                                          void* data,
                                          hid_t type_id,
                                          size_t size) const {
    if (!HasAttribute(name)) {
        return absl::NotFoundError(
            tensorstore::StrCat("Attribute not found: ", name));
    }
    
    hid_t attr_id = H5Aopen(object_id_, name.c_str(), H5P_DEFAULT);
    if (attr_id < 0) {
        return absl::InternalError(
            tensorstore::StrCat("Failed to open attribute: ", name));
    }
    
    herr_t status = H5Aread(attr_id, type_id, data);
    H5Aclose(attr_id);
    
    if (status < 0) {
        return absl::InternalError(
            tensorstore::StrCat("Failed to read attribute: ", name));
    }
    
    return absl::OkStatus();
}

bool AttributeManager::HasAttribute(const std::string& name) const {
    return H5Aexists(object_id_, name.c_str()) > 0;
}

Result<void> AttributeManager::WriteJsonAttribute(const std::string& name,
                                               const nlohmann::json& value) {
    TENSORSTORE_ASSIGN_OR_RETURN(auto [type_id, data], ConvertJsonToHDF5(value));
    return WriteAttribute(name, data.data(), type_id, data.size());
}

Result<nlohmann::json> AttributeManager::ReadJsonAttribute(
    const std::string& name) const {
    if (!HasAttribute(name)) {
        return absl::NotFoundError(
            tensorstore::StrCat("Attribute not found: ", name));
    }
    
    hid_t attr_id = H5Aopen(object_id_, name.c_str(), H5P_DEFAULT);
    if (attr_id < 0) {
        return absl::InternalError(
            tensorstore::StrCat("Failed to open attribute: ", name));
    }
    
    // Get attribute type and size
    hid_t type_id = H5Aget_type(attr_id);
    hid_t space_id = H5Aget_space(attr_id);
    hsize_t size;
    H5Sget_simple_extent_dims(space_id, &size, nullptr);
    
    // Read data
    std::vector<unsigned char> data(size);
    herr_t status = H5Aread(attr_id, type_id, data.data());
    
    H5Sclose(space_id);
    H5Aclose(attr_id);
    
    if (status < 0) {
        return absl::InternalError(
            tensorstore::StrCat("Failed to read attribute: ", name));
    }
    
    return ConvertHDF5ToJson(type_id, data.data(), size);
}

std::vector<std::string> AttributeManager::ListAttributes() const {
    std::vector<std::string> result;
    
    hsize_t num_attrs = H5Aget_num_attrs(object_id_);
    result.reserve(num_attrs);
    
    for (hsize_t i = 0; i < num_attrs; ++i) {
        char name[256];
        H5Aget_name_by_idx(object_id_, ".", H5_INDEX_NAME, H5_ITER_NATIVE,
                          i, name, sizeof(name), H5P_DEFAULT);
        result.push_back(name);
    }
    
    return result;
}

Result<std::pair<hid_t, std::vector<unsigned char>>> AttributeManager::ConvertJsonToHDF5(
    const nlohmann::json& value) const {
    std::vector<unsigned char> data;
    hid_t type_id;
    
    switch (value.type()) {
        case nlohmann::json::value_t::number_integer:
            type_id = H5T_NATIVE_INT64;
            data.resize(sizeof(int64_t));
            *reinterpret_cast<int64_t*>(data.data()) = value.get<int64_t>();
            break;
            
        case nlohmann::json::value_t::number_float:
            type_id = H5T_NATIVE_DOUBLE;
            data.resize(sizeof(double));
            *reinterpret_cast<double*>(data.data()) = value.get<double>();
            break;
            
        case nlohmann::json::value_t::string: {
            std::string str = value.get<std::string>();
            type_id = H5Tcopy(H5T_C_S1);
            H5Tset_size(type_id, str.size());
            data.resize(str.size());
            std::memcpy(data.data(), str.data(), str.size());
            break;
        }
            
        default:
            return absl::InvalidArgumentError("Unsupported JSON type");
    }
    
    return std::make_pair(type_id, std::move(data));
}

Result<nlohmann::json> AttributeManager::ConvertHDF5ToJson(hid_t type_id,
                                                        const void* data,
                                                        size_t size) const {
    H5T_class_t type_class = H5Tget_class(type_id);
    
    switch (type_class) {
        case H5T_INTEGER:
            return nlohmann::json(*static_cast<const int64_t*>(data));
            
        case H5T_FLOAT:
            return nlohmann::json(*static_cast<const double*>(data));
            
        case H5T_STRING:
            return nlohmann::json(std::string(static_cast<const char*>(data), size));
            
        default:
            return absl::InvalidArgumentError("Unsupported HDF5 type");
    }
}

}  // namespace hdf5_driver
}  // namespace tensorstore
