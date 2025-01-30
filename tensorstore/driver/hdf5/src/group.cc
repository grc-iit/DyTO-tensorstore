#include "tensorstore/driver/hdf5/group.h"

#include "absl/status/status.h"
#include "tensorstore/util/str_cat.h"

namespace tensorstore {
namespace hdf5_driver {

Result<HDF5Group> HDF5Group::Create(hid_t file_id,
                                  const std::string& path) {
    // Create intermediate groups if needed
    hid_t gcpl = H5Pcreate(H5P_GROUP_CREATE);
    H5Pset_create_intermediate_group(gcpl, 1);
    
    hid_t group_id = H5Gcreate2(file_id, path.c_str(),
                               gcpl, H5P_DEFAULT, H5P_DEFAULT);
    H5Pclose(gcpl);
    
    if (group_id < 0) {
        return absl::InternalError(
            tensorstore::StrCat("Failed to create group: ", path));
    }
    
    return HDF5Group(group_id);
}

Result<HDF5Group> HDF5Group::Open(hid_t file_id,
                                const std::string& path) {
    hid_t group_id = H5Gopen2(file_id, path.c_str(), H5P_DEFAULT);
    if (group_id < 0) {
        return absl::NotFoundError(
            tensorstore::StrCat("Group not found: ", path));
    }
    
    return HDF5Group(group_id);
}

Result<std::vector<std::string>> HDF5Group::ListChildren() const {
    std::vector<std::string> result;
    
    // Get number of objects in group
    H5G_info_t group_info;
    herr_t status = H5Gget_info(group_id_, &group_info);
    if (status < 0) {
        return absl::InternalError("Failed to get group info");
    }
    
    result.reserve(group_info.nlinks);
    
    // Iterate over objects
    for (hsize_t i = 0; i < group_info.nlinks; ++i) {
        char name[256];
        H5Lget_name_by_idx(group_id_, ".", H5_INDEX_NAME, H5_ITER_NATIVE,
                          i, name, sizeof(name), H5P_DEFAULT);
        result.push_back(name);
    }
    
    return result;
}

Result<bool> HDF5Group::HasChild(const std::string& name) const {
    return H5Lexists(group_id_, name.c_str(), H5P_DEFAULT) > 0;
}

Result<void> HDF5Group::DeleteChild(const std::string& name) {
    TENSORSTORE_ASSIGN_OR_RETURN(bool exists, HasChild(name));
    if (!exists) {
        return absl::NotFoundError(
            tensorstore::StrCat("Child not found: ", name));
    }
    
    herr_t status = H5Ldelete(group_id_, name.c_str(), H5P_DEFAULT);
    if (status < 0) {
        return absl::InternalError(
            tensorstore::StrCat("Failed to delete child: ", name));
    }
    
    return absl::OkStatus();
}

}  // namespace hdf5_driver
}  // namespace tensorstore
