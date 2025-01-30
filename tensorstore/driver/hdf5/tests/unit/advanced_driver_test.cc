#include "tensorstore/driver/hdf5/driver.h"

#include <memory>
#include <vector>
#include <string>

#include "absl/status/status.h"
#include "gtest/gtest.h"
#include "tensorstore/context.h"
#include "tensorstore/data_type.h"
#include "tensorstore/driver/hdf5/metadata.h"
#include "tensorstore/driver/hdf5/schema.h"
#include "tensorstore/index.h"
#include "tensorstore/util/result.h"
#include "tensorstore/util/status_testutil.h"
#include "tensorstore/util/temp_directory.h"

namespace {

using ::tensorstore::DataType;
using ::tensorstore::DimensionIndex;
using ::tensorstore::Index;
using ::tensorstore::MatchesStatus;
using ::tensorstore::Result;
using ::tensorstore::TempDir;
using ::tensorstore::hdf5_driver::HDF5Driver;
using ::tensorstore::hdf5_driver::HDF5Metadata;

class HDF5AdvancedDriverTest : public ::testing::Test {
protected:
    void SetUp() override {
        temp_dir_ = std::make_unique<TempDir>();
        test_path_ = temp_dir_->path() / "test.h5";
    }
    
    void TearDown() override {
        temp_dir_.reset();
    }
    
    std::unique_ptr<TempDir> temp_dir_;
    std::string test_path_;
};

Result<std::unique_ptr<HDF5Driver>> CreateTestDriver(
    const TempDir& temp_dir,
    const Schema& schema,
    const CompressionParams& compression = {}) {
    auto driver = std::make_unique<HDF5Driver>();
    
    // Configure driver with schema
    TENSORSTORE_RETURN_IF_ERROR(driver->Initialize(schema));
    
    // Set file path
    std::string file_path = temp_dir.path() / "test.h5";
    TENSORSTORE_RETURN_IF_ERROR(driver->SetFilePath(file_path));
    
    // Apply compression if specified
    if (!compression.empty()) {
        TENSORSTORE_RETURN_IF_ERROR(driver->SetCompression(compression));
    }
    
    return driver;
}

TEST_F(HDF5AdvancedDriverTest, CreateAndOpen) {
    Schema schema;
    schema.dtype(DataType::Float32())
          .rank(2)
          .shape({100, 200});
    
    TENSORSTORE_ASSERT_OK_AND_ASSIGN(
        auto driver,
        CreateTestDriver(*temp_dir_, schema));
    
    EXPECT_EQ(driver->dtype(), DataType::Float32());
    EXPECT_EQ(driver->rank(), 2);
    
    // Verify dimensions
    auto shape = driver->shape();
    ASSERT_EQ(shape.size(), 2);
    EXPECT_EQ(shape[0], 100);
    EXPECT_EQ(shape[1], 200);
}

TEST_F(HDF5AdvancedDriverTest, ReadWrite) {
    Schema schema;
    schema.dtype(DataType::Float32())
          .rank(2)
          .shape({10, 10});
    
    TENSORSTORE_ASSERT_OK_AND_ASSIGN(
        auto driver,
        CreateTestDriver(*temp_dir_, schema));
    
    // Create test data
    std::vector<float> write_data(100, 1.5f);
    
    // Write data
    auto write_status = driver->Write(write_data.data(), {0, 0}, {10, 10});
    EXPECT_TRUE(write_status.ok()) << write_status.ToString();
    
    // Read data back
    std::vector<float> read_data(100);
    auto read_status = driver->Read(read_data.data(), {0, 0}, {10, 10});
    EXPECT_TRUE(read_status.ok()) << read_status.ToString();
    
    // Verify data
    for (size_t i = 0; i < write_data.size(); ++i) {
        EXPECT_FLOAT_EQ(write_data[i], read_data[i])
            << "Data mismatch at index " << i;
    }
}

}  // namespace
