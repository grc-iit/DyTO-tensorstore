#include <memory>
#include <vector>
#include <string>

#include "gtest/gtest.h"
#include "tensorstore/driver/hdf5/driver.h"
#include "tensorstore/driver/hdf5/metadata.h"
#include "tensorstore/data_type.h"
#include "tensorstore/index.h"
#include "tensorstore/util/status_testutil.h"
#include "nlohmann/json.hpp"

namespace {
using ::tensorstore::DataType;
using ::tensorstore::DimensionIndex;
using ::tensorstore::Index;
using ::tensorstore::MatchesStatus;
using ::tensorstore::Result;
using ::tensorstore::hdf5_driver::HDF5Driver;
using ::tensorstore::hdf5_driver::HDF5Metadata;

class HDF5AdvancedFeaturesTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Create test file
        filename_ = "advanced_test.h5";
        file_id_ = H5Fcreate(filename_.c_str(), H5F_ACC_TRUNC,
                            H5P_DEFAULT, H5P_DEFAULT);
        ASSERT_GE(file_id_, 0);

        // Set up base metadata
        metadata_.rank = 2;
        metadata_.dtype = tensorstore::dtype_v<float>;
        metadata_.shape = {100, 100};  // Large enough for compression
        metadata_.chunks = {20, 20};
        metadata_.dimension_labels = {"x", "y"};
        metadata_.h5_type = H5T_NATIVE_FLOAT;
    }

    void TearDown() override {
        if (file_id_ >= 0) {
            H5Fclose(file_id_);
        }
        std::remove(filename_.c_str());
    }

    std::unique_ptr<HDF5Driver> CreateCompressedDriver(int level = 6) {
        // Create dataset with compression
        hid_t dcpl = H5Pcreate(H5P_DATASET_CREATE);
        H5Pset_chunk(dcpl, metadata_.rank, metadata_.chunks.data());
        H5Pset_deflate(dcpl, level);

        std::vector<hsize_t> dims(metadata_.shape.begin(), metadata_.shape.end());
        hid_t space_id = H5Screate_simple(metadata_.rank, dims.data(), nullptr);

        hid_t dataset_id = H5Dcreate2(file_id_, "/compressed_dataset",
                                     metadata_.h5_type, space_id,
                                     H5P_DEFAULT, dcpl, H5P_DEFAULT);

        H5Pclose(dcpl);
        H5Sclose(space_id);

        metadata_.dataset_id = dataset_id;
        return std::make_unique<HDF5Driver>(dataset_id, metadata_);
    }

    std::unique_ptr<HDF5Driver> CreateAttributeTestDriver() {
        std::vector<hsize_t> dims(metadata_.shape.begin(), metadata_.shape.end());
        hid_t space_id = H5Screate_simple(metadata_.rank, dims.data(), nullptr);

        hid_t dataset_id = H5Dcreate2(file_id_, "/attribute_dataset",
                                     metadata_.h5_type, space_id,
                                     H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

        H5Sclose(space_id);

        metadata_.dataset_id = dataset_id;
        return std::make_unique<HDF5Driver>(dataset_id, metadata_);
    }

    // Generate test data with a repeating pattern for better compression
    std::vector<float> GenerateTestData() {
        std::vector<float> data(metadata_.shape[0] * metadata_.shape[1]);
        for (size_t i = 0; i < data.size(); ++i) {
            data[i] = static_cast<float>(i % 10);  // Repeating pattern
        }
        return data;
    }

    std::string filename_;
    hid_t file_id_ = -1;
    HDF5Metadata metadata_;
};

TEST_F(HDF5AdvancedFeaturesTest, Compression) {
    auto driver = CreateCompressedDriver();
    auto test_data = GenerateTestData();

    // Write data
    tensorstore::WriteRequest write_request;
    write_request.transform = tensorstore::IdentityTransform(metadata_.rank);
    
    bool write_completed = false;
    driver->Write(std::move(write_request),
                 [&](tensorstore::WriteChunk chunk) {
                     if (!chunk.ok()) {
                         ADD_FAILURE() << chunk.status();
                         return;
                     }
                     chunk.data()->data.resize(test_data.size() * sizeof(float));
                     std::memcpy(chunk.data()->data.data(), test_data.data(),
                               test_data.size() * sizeof(float));
                     write_completed = true;
                 });
    
    EXPECT_TRUE(write_completed);

    // Read data back
    tensorstore::ReadRequest read_request;
    read_request.transform = tensorstore::IdentityTransform(metadata_.rank);
    
    bool read_completed = false;
    std::vector<float> read_data(test_data.size());
    
    driver->Read(std::move(read_request),
                [&](tensorstore::ReadChunk chunk) {
                    if (!chunk.ok()) {
                        ADD_FAILURE() << chunk.status();
                        return;
                    }
                    std::memcpy(read_data.data(), chunk.data().data(),
                              chunk.data().size());
                    read_completed = true;
                });
    
    EXPECT_TRUE(read_completed);
    EXPECT_EQ(read_data, test_data);

    // Verify compression
    hid_t dataset_id = metadata_.dataset_id;
    hid_t plist = H5Dget_create_plist(dataset_id);
    
    unsigned int filter_info;
    H5Zget_filter_info(H5Z_FILTER_DEFLATE, &filter_info);
    EXPECT_TRUE(filter_info & H5Z_FILTER_CONFIG_ENCODE_ENABLED);
    
    H5Pclose(plist);
}

TEST_F(HDF5AdvancedFeaturesTest, Attributes) {
    auto driver = CreateAttributeTestDriver();

    // Test various attribute types
    {
        nlohmann::json metadata = {
            {"description", "Test dataset"},
            {"created", "2025-01-30"},
            {"version", 1},
            {"parameters", {
                {"min_value", 0.0},
                {"max_value", 100.0},
                {"flags", {true, false, true}}
            }}
        };

        EXPECT_TRUE(driver->WriteMetadata("info", metadata).ok());

        auto read_result = driver->ReadMetadata("info");
        EXPECT_TRUE(read_result.ok());
        EXPECT_EQ(*read_result, metadata);
    }

    // Test attribute listing
    {
        auto attrs = driver->ListMetadata();
        EXPECT_EQ(attrs.size(), 1);
        EXPECT_EQ(attrs[0], "info");
    }

    // Test error cases
    {
        // Reading non-existent attribute
        auto result = driver->ReadMetadata("nonexistent");
        EXPECT_THAT(result, MatchesStatus(absl::StatusCode::kNotFound));

        // Writing invalid JSON
        nlohmann::json invalid = nullptr;
        auto write_result = driver->WriteMetadata("invalid", invalid);
        EXPECT_FALSE(write_result.ok());
    }
}

TEST_F(HDF5AdvancedFeaturesTest, CompressionLevels) {
    // Test different compression levels
    std::vector<int> levels = {1, 3, 6, 9};
    auto test_data = GenerateTestData();
    std::vector<size_t> sizes;

    for (int level : levels) {
        auto driver = CreateCompressedDriver(level);

        // Write data
        tensorstore::WriteRequest write_request;
        write_request.transform = tensorstore::IdentityTransform(metadata_.rank);
        
        driver->Write(std::move(write_request),
                     [&](tensorstore::WriteChunk chunk) {
                         chunk.data()->data.resize(test_data.size() * sizeof(float));
                         std::memcpy(chunk.data()->data.data(), test_data.data(),
                                   test_data.size() * sizeof(float));
                     });

        // Get dataset size
        hsize_t size;
        H5Dget_storage_size(metadata_.dataset_id, &size);
        sizes.push_back(size);
    }

    // Verify that higher compression levels generally result in smaller sizes
    for (size_t i = 1; i < sizes.size(); ++i) {
        EXPECT_LE(sizes[i], sizes[i-1]);
    }
}

}  // namespace
