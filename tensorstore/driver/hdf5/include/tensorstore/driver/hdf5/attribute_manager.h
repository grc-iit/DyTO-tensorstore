#ifndef TENSORSTORE_DRIVER_HDF5_ATTRIBUTE_MANAGER_H_
#define TENSORSTORE_DRIVER_HDF5_ATTRIBUTE_MANAGER_H_

#include <string>
#include <vector>

#include "absl/status/status.h"
#include "tensorstore/util/result.h"
#include "tensorstore/json_serialization_options.h"
#include "nlohmann/json.hpp"
#include "hdf5.h"

namespace tensorstore {
namespace hdf5_driver {

/// Manages HDF5 attributes for datasets and groups
class AttributeManager {
public:
    /// Constructor
    /// @param object_id HDF5 object identifier (dataset or group)
    explicit AttributeManager(hid_t object_id) : object_id_(object_id) {}

    /// Write an attribute
    /// @param name Attribute name
    /// @param data Pointer to data to write
    /// @param type_id HDF5 datatype identifier
    /// @param size Size of data in bytes
    Result<void> WriteAttribute(const std::string& name,
                              const void* data,
                              hid_t type_id,
                              size_t size);
    
    /// Read an attribute
    /// @param name Attribute name
    /// @param data Buffer to store data
    /// @param type_id HDF5 datatype identifier
    /// @param size Size of buffer in bytes
    Result<void> ReadAttribute(const std::string& name,
                             void* data,
                             hid_t type_id,
                             size_t size) const;
    
    /// Check if attribute exists
    /// @param name Attribute name
    /// @return True if attribute exists
    bool HasAttribute(const std::string& name) const;

    /// Write JSON value as attribute
    /// @param name Attribute name
    /// @param value JSON value to write
    Result<void> WriteJsonAttribute(const std::string& name,
                                  const nlohmann::json& value);

    /// Read attribute as JSON value
    /// @param name Attribute name
    /// @return JSON value
    Result<nlohmann::json> ReadJsonAttribute(const std::string& name) const;

    /// List all attributes
    /// @return Vector of attribute names
    std::vector<std::string> ListAttributes() const;

private:
    /// Convert JSON value to HDF5 type and data
    /// @param value JSON value to convert
    /// @return Pair of HDF5 type ID and serialized data
    Result<std::pair<hid_t, std::vector<unsigned char>>> ConvertJsonToHDF5(
        const nlohmann::json& value) const;

    /// Convert HDF5 data to JSON value
    /// @param type_id HDF5 datatype identifier
    /// @param data Raw data buffer
    /// @param size Size of data in bytes
    /// @return JSON value
    Result<nlohmann::json> ConvertHDF5ToJson(hid_t type_id,
                                            const void* data,
                                            size_t size) const;

    hid_t object_id_;  ///< HDF5 object identifier
};

}  // namespace hdf5_driver
}  // namespace tensorstore

#endif  // TENSORSTORE_DRIVER_HDF5_ATTRIBUTE_MANAGER_H_
