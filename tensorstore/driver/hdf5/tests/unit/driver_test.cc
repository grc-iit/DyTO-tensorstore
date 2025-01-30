#include "tensorstore/driver/hdf5/driver.h"

#include <memory>
#include <vector>

#include "absl/status/status.h"
#include "gtest/gtest.h"
#include "tensorstore/context.h"
#include "tensorstore/data_type.h"
#include "tensorstore/driver/hdf5/metadata.h"
#include "tensorstore/driver/hdf5/schema.h"
#include "tensorstore/index.h"
#include "tensorstore/util/result.h"
#include "tensorstore/util/status_testutil.h"

namespace {
using ::tensorstore::DataType;
using ::tensorstore::DimensionIndex;
using ::tensorstore::Index;
using ::tensorstore::MatchesStatus;
using ::tensorstore::Result;
using ::tensorstore::hdf5_driver::HDF5Driver;
using ::tensorstore::hdf5_driver::HDF5Metadata;

class HDF5DriverTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Create a temporary HDF5 file for testing
        std::string filename = "test_dataset.h5";
        hid_t file_id = H5Fcreate(filename.c_str(), H5F_ACC_TRUNC, 
                                 H5P_DEFAULT, H5P_DEFAULT);
        ASSERT_GE(file_id, 0);

        // Create test dataset
        std::vector<hsize_t> dims = {4, 6};  // 4x6 dataset
        std::vector<hsize_t> chunk_dims = {2, 3};  // 2x3 chunks
        hid_t dataspace = H5Screate_simple(2, dims.data(), nullptr);
        ASSERT_GE(dataspace, 0);

        // Create chunked dataset
        hid_t dcpl = H5Pcreate(H5P_DATASET_CREATE);
        H5Pset_chunk(dcpl, 2, chunk_dims.data());
        
        dataset_id_ = H5Dcreate2(file_id, "/test_dataset", H5T_NATIVE_FLOAT,
                                dataspace, H5P_DEFAULT, dcpl, H5P_DEFAULT);
        ASSERT_GE(dataset_id_, 0);

        // Clean up
        H5Pclose(dcpl);
        H5Sclose(dataspace);
        H5Fclose(file_id);

        // Set up metadata
        metadata_.rank = 2;
        metadata_.dtype = tensorstore::dtype_v<float>;
        metadata_.shape = {4, 6};
        metadata_.chunks = {2, 3};
        metadata_.dimension_labels = {"x", "y"};
        metadata_.h5_type = H5T_NATIVE_FLOAT;
    }

    void TearDown() override {
        if (dataset_id_ >= 0) {
            H5Dclose(dataset_id_);
        }
        // Delete the test file
        std::remove("test_dataset.h5");
    }

    std::unique_ptr<HDF5Driver> CreateTestDriver() {
        return std::make_unique<HDF5Driver>(dataset_id_, metadata_);
    }

    hid_t dataset_id_ = -1;
    HDF5Metadata metadata_;
};

TEST_F(HDF5DriverTest, ReadData) {
    auto driver = CreateTestDriver();
    
    // Write some test data first
    std::vector<float> write_data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    herr_t status = H5Dwrite(dataset_id_, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL,
                            H5P_DEFAULT, write_data.data());
    ASSERT_GE(status, 0);

    // Create read request
    tensorstore::ReadRequest request;
    request.transform = tensorstore::IdentityTransform(metadata_.rank);
    
    bool received_data = false;
    driver->Read(std::move(request),
                [&](tensorstore::ReadChunk chunk) {
                    if (!chunk.ok()) {
                        ADD_FAILURE() << chunk.status();
                        return;
                    }
                    received_data = true;
                    auto& data = chunk.data();
                    EXPECT_EQ(data.size(), write_data.size());
                    // Compare data
                    const float* read_ptr = reinterpret_cast<const float*>(data.data());
                    for (size_t i = 0; i < write_data.size(); ++i) {
                        EXPECT_FLOAT_EQ(read_ptr[i], write_data[i]);
                    }
                });
    
    EXPECT_TRUE(received_data);
}

TEST_F(HDF5DriverTest, WriteData) {
    auto driver = CreateTestDriver();
    
    // Create write request
    tensorstore::WriteRequest request;
    request.transform = tensorstore::IdentityTransform(metadata_.rank);
    
    std::vector<float> write_data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    auto chunk_data = std::make_shared<tensorstore::ChunkData>();
    chunk_data->data.resize(write_data.size() * sizeof(float));
    std::memcpy(chunk_data->data.data(), write_data.data(), 
                write_data.size() * sizeof(float));
    
    bool completed = false;
    driver->Write(std::move(request),
                 [&](tensorstore::WriteChunk chunk) {
                     if (!chunk.ok()) {
                         ADD_FAILURE() << chunk.status();
                         return;
                     }
                     chunk.data() = chunk_data;
                     completed = true;
                 });
    
    EXPECT_TRUE(completed);
    
    // Verify written data
    std::vector<float> read_data(write_data.size());
    herr_t status = H5Dread(dataset_id_, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL,
                           H5P_DEFAULT, read_data.data());
    ASSERT_GE(status, 0);
    
    for (size_t i = 0; i < write_data.size(); ++i) {
        EXPECT_FLOAT_EQ(read_data[i], write_data[i]);
    }
}

TEST_F(HDF5DriverTest, TypeConversion) {
    // Test integer types
    EXPECT_EQ(ConvertHDF5Type(H5T_NATIVE_INT8).value(), 
              tensorstore::dtype_v<std::int8_t>);
    EXPECT_EQ(ConvertHDF5Type(H5T_NATIVE_UINT8).value(),
              tensorstore::dtype_v<std::uint8_t>);
    EXPECT_EQ(ConvertHDF5Type(H5T_NATIVE_INT16).value(),
              tensorstore::dtype_v<std::int16_t>);
    EXPECT_EQ(ConvertHDF5Type(H5T_NATIVE_UINT16).value(),
              tensorstore::dtype_v<std::uint16_t>);
    EXPECT_EQ(ConvertHDF5Type(H5T_NATIVE_INT32).value(),
              tensorstore::dtype_v<std::int32_t>);
    EXPECT_EQ(ConvertHDF5Type(H5T_NATIVE_UINT32).value(),
              tensorstore::dtype_v<std::uint32_t>);
    EXPECT_EQ(ConvertHDF5Type(H5T_NATIVE_INT64).value(),
              tensorstore::dtype_v<std::int64_t>);
    EXPECT_EQ(ConvertHDF5Type(H5T_NATIVE_UINT64).value(),
              tensorstore::dtype_v<std::uint64_t>);
    
    // Test floating point types
    EXPECT_EQ(ConvertHDF5Type(H5T_NATIVE_FLOAT).value(),
              tensorstore::dtype_v<float>);
    EXPECT_EQ(ConvertHDF5Type(H5T_NATIVE_DOUBLE).value(),
              tensorstore::dtype_v<double>);
    
    // Test boolean type
    EXPECT_EQ(ConvertHDF5Type(H5T_NATIVE_HBOOL).value(),
              tensorstore::dtype_v<bool>);
    
    // Test unsupported type
    EXPECT_THAT(ConvertHDF5Type(H5T_NATIVE_OPAQUE),
                MatchesStatus(absl::StatusCode::kInvalidArgument));
}

TEST_F(HDF5DriverTest, GetSchema) {
    auto driver = CreateTestDriver();
    TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto schema, driver->GetSchema());
    
    EXPECT_EQ(schema.rank(), metadata_.rank);
    EXPECT_EQ(schema.dtype(), metadata_.dtype);
    
    auto dims = schema.dimension_constraints();
    ASSERT_TRUE(dims.valid());
    for (DimensionIndex i = 0; i < metadata_.rank; ++i) {
        EXPECT_EQ(dims[i].inclusive_min(), 0);
        EXPECT_EQ(dims[i].exclusive_max(), metadata_.shape[i]);
        EXPECT_EQ(dims[i].label(), metadata_.dimension_labels[i]);
    }
}

TEST_F(HDF5DriverTest, GetChunkLayout) {
    auto driver = CreateTestDriver();
    TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto layout, driver->GetChunkLayout());
    
    EXPECT_EQ(layout.rank(), metadata_.rank);
    
    auto chunk_shape = layout.chunk_shape();
    ASSERT_TRUE(chunk_shape.valid());
    for (DimensionIndex i = 0; i < metadata_.rank; ++i) {
        EXPECT_EQ(chunk_shape[i], metadata_.chunks[i]);
    }
    
    auto grid_origin = layout.grid_origin();
    ASSERT_TRUE(grid_origin.valid());
    for (DimensionIndex i = 0; i < metadata_.rank; ++i) {
        EXPECT_EQ(grid_origin[i], 0);
    }
}

}  // namespace
