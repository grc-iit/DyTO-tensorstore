#include "tensorstore/driver/hdf5/driver.h"

#include <memory>
#include <vector>
#include <string>

#include "absl/status/status.h"
#include "absl/time/time.h"
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

// Test utilities
std::unique_ptr<HDF5Driver> CreateTestDataset() {
    Schema schema;
    schema.dtype(DataType::Float32())
          .rank(3)
          .shape({100, 100, 100})
          .chunk_layout({10, 10, 10});
    
    auto temp_dir = std::make_unique<TempDir>();
    std::string file_path = temp_dir->path() / "test.h5";
    
    auto driver = std::make_unique<HDF5Driver>();
    TENSORSTORE_CHECK_OK(driver->Initialize(schema));
    TENSORSTORE_CHECK_OK(driver->SetFilePath(file_path));
    
    return driver;
}

void WriteTestData(HDF5Driver* driver) {
    const size_t size = 100 * 100 * 100;
    std::vector<float> data(size);
    for (size_t i = 0; i < size; ++i) {
        data[i] = static_cast<float>(i);
    }
    
    TENSORSTORE_CHECK_OK(driver->Write(data.data(), {0, 0, 0}, {100, 100, 100}));
}

void VerifyTestData(HDF5Driver* driver) {
    const size_t size = 100 * 100 * 100;
    std::vector<float> data(size);
    
    TENSORSTORE_CHECK_OK(driver->Read(data.data(), {0, 0, 0}, {100, 100, 100}));
    
    for (size_t i = 0; i < size; ++i) {
        EXPECT_FLOAT_EQ(data[i], static_cast<float>(i))
            << "Data mismatch at index " << i;
    }
}

void ModifyAndVerifyAttributes(HDF5Driver* driver) {
    // Set attributes
    TENSORSTORE_CHECK_OK(driver->SetAttribute("description", "Test dataset"));
    TENSORSTORE_CHECK_OK(driver->SetAttribute("version", 1.0f));
    
    // Verify attributes
    std::string description;
    TENSORSTORE_CHECK_OK(driver->GetAttribute("description", &description));
    EXPECT_EQ(description, "Test dataset");
    
    float version;
    TENSORSTORE_CHECK_OK(driver->GetAttribute("version", &version));
    EXPECT_FLOAT_EQ(version, 1.0f);
}

void TestGroupOperations(HDF5Driver* driver) {
    // Create a group
    TENSORSTORE_CHECK_OK(driver->CreateGroup("/test_group"));
    
    // Create a dataset in the group
    Schema sub_schema;
    sub_schema.dtype(DataType::Int32())
             .rank(1)
             .shape({10});
    
    TENSORSTORE_CHECK_OK(driver->CreateDataset("/test_group/data", sub_schema));
    
    // Write data to the dataset
    std::vector<int32_t> data(10);
    for (int i = 0; i < 10; ++i) data[i] = i;
    
    TENSORSTORE_CHECK_OK(driver->Write(data.data(), {0}, {10}, "/test_group/data"));
    
    // Read and verify data
    std::vector<int32_t> read_data(10);
    TENSORSTORE_CHECK_OK(driver->Read(read_data.data(), {0}, {10}, "/test_group/data"));
    
    for (int i = 0; i < 10; ++i) {
        EXPECT_EQ(read_data[i], i);
    }
}

TEST(HDF5IntegrationTest, CompleteWorkflow) {
    auto driver = CreateTestDataset();
    ASSERT_NE(driver, nullptr);
    
    WriteTestData(driver.get());
    VerifyTestData(driver.get());
    ModifyAndVerifyAttributes(driver.get());
    TestGroupOperations(driver.get());
}

}  // namespace
