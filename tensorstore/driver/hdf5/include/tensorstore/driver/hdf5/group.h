#ifndef TENSORSTORE_DRIVER_HDF5_GROUP_H_
#define TENSORSTORE_DRIVER_HDF5_GROUP_H_

#include <string>
#include <vector>

#include "absl/status/status.h"
#include "tensorstore/util/result.h"
#include "hdf5.h"

namespace tensorstore {
namespace hdf5_driver {

/// Manages HDF5 groups
class HDF5Group {
public:
    /// Create a new group
    /// @param file_id HDF5 file identifier
    /// @param path Path to the group
    /// @return New group instance
    static Result<HDF5Group> Create(hid_t file_id,
                                  const std::string& path);
    
    /// Open an existing group
    /// @param file_id HDF5 file identifier
    /// @param path Path to the group
    /// @return Group instance
    static Result<HDF5Group> Open(hid_t file_id,
                                const std::string& path);
    
    /// Constructor
    explicit HDF5Group(hid_t group_id) : group_id_(group_id) {}
    
    /// Destructor
    ~HDF5Group() {
        if (group_id_ >= 0) {
            H5Gclose(group_id_);
        }
    }
    
    // Disable copy
    HDF5Group(const HDF5Group&) = delete;
    HDF5Group& operator=(const HDF5Group&) = delete;
    
    // Enable move
    HDF5Group(HDF5Group&& other) noexcept
        : group_id_(std::exchange(other.group_id_, -1)) {}
    HDF5Group& operator=(HDF5Group&& other) noexcept {
        if (this != &other) {
            if (group_id_ >= 0) {
                H5Gclose(group_id_);
            }
            group_id_ = std::exchange(other.group_id_, -1);
        }
        return *this;
    }
    
    /// List all children in the group
    /// @return Vector of child names
    Result<std::vector<std::string>> ListChildren() const;
    
    /// Check if a child exists
    /// @param name Child name
    /// @return True if child exists
    Result<bool> HasChild(const std::string& name) const;
    
    /// Delete a child
    /// @param name Child name
    Result<void> DeleteChild(const std::string& name);
    
    /// Get group ID
    hid_t id() const { return group_id_; }

private:
    hid_t group_id_;  ///< HDF5 group identifier
};

}  // namespace hdf5_driver
}  // namespace tensorstore

#endif  // TENSORSTORE_DRIVER_HDF5_GROUP_H_
