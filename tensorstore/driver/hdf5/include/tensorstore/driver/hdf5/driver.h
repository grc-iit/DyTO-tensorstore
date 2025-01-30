#ifndef TENSORSTORE_DRIVER_HDF5_DRIVER_H_
#define TENSORSTORE_DRIVER_HDF5_DRIVER_H_

#include "tensorstore/driver/driver.h"
#include "tensorstore/driver/registry.h"
#include "tensorstore/index.h"
#include "tensorstore/rank.h"
#include "tensorstore/schema.h"
#include "tensorstore/data_type.h"
#include "tensorstore/driver/chunk.h"
#include "tensorstore/driver/hdf5/metadata.h"

namespace tensorstore {
namespace hdf5_driver {

class HDF5Cache;  // Forward declaration

/// HDF5 driver implementation
class HDF5Driver : public RegisteredDriver<HDF5Driver> {
 public:
  static constexpr const char id[] = "hdf5";
  
  // Core driver interface implementations
  DataType dtype() override;
  DimensionIndex rank() override;
  
  Result<Schema> GetSchema() override;
  Result<ChunkLayout> GetChunkLayout() override;
  
  void Read(ReadRequest request, ReadChunkReceiver receiver) override;
  void Write(WriteRequest request, WriteChunkReceiver receiver) override;

  /// Write metadata attribute
  /// @param key Attribute name
  /// @param value JSON value to write
  Result<void> WriteMetadata(const std::string& key,
                             const nlohmann::json& value);

  /// Read metadata attribute
  /// @param key Attribute name
  /// @return JSON value
  Result<nlohmann::json> ReadMetadata(const std::string& key);

  /// List all metadata attributes
  /// @return Vector of attribute names
  std::vector<std::string> ListMetadata() const;

  /// Open a dataset in the HDF5 file
  /// @param path Path to the dataset
  /// @return Driver handle for the dataset
  static Result<std::shared_ptr<HDF5Driver>> OpenDataset(
      hid_t file_id, const std::string& path);

  /// Create a dataset in the HDF5 file
  /// @param path Path to the dataset
  /// @param metadata Dataset metadata
  /// @return Driver handle for the dataset
  static Result<std::shared_ptr<HDF5Driver>> CreateDataset(
      hid_t file_id, const std::string& path,
      const HDF5Metadata& metadata);

 private:
  std::shared_ptr<HDF5Cache> cache_;
  HDF5Metadata metadata_;

  /// Helper to extract group path from dataset path
  static std::string GetGroupPath(const std::string& path);
  
  /// Helper to extract dataset name from path
  static std::string GetBaseName(const std::string& path);
  
  /// Open a dataset in a group
  static Result<std::shared_ptr<HDF5Driver>> OpenDatasetInGroup(
      const HDF5Group& group, const std::string& name);
};

}  // namespace hdf5_driver
}  // namespace tensorstore

#endif  // TENSORSTORE_DRIVER_HDF5_DRIVER_H_
