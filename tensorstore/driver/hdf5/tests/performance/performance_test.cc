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
#include "tensorstore/internal/metrics/registry.h"
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

std::unique_ptr<HDF5Driver> CreateLargeTestDataset(size_t size_bytes) {
    // Calculate dimensions for a cubic dataset
    size_t dim_size = static_cast<size_t>(std::cbrt(size_bytes / sizeof(float)));
    
    Schema schema;
    schema.dtype(DataType::Float32())
          .rank(3)
          .shape({dim_size, dim_size, dim_size})
          .chunk_layout({64, 64, 64});  // 1MB chunks (64^3 * 4 bytes)
    
    auto temp_dir = std::make_unique<TempDir>();
    std::string file_path = temp_dir->path() / "large_test.h5";
    
    auto driver = std::make_unique<HDF5Driver>();
    TENSORSTORE_CHECK_OK(driver->Initialize(schema));
    TENSORSTORE_CHECK_OK(driver->SetFilePath(file_path));
    
    // Enable compression
    CompressionParams compression;
    compression.method = "gzip";
    compression.level = 1;  // Light compression for better write performance
    TENSORSTORE_CHECK_OK(driver->SetCompression(compression));
    
    return driver;
}

void RecordBenchmarkResult(const std::string& metric_name,
                          absl::Duration duration,
                          size_t bytes_processed) {
    double seconds = absl::ToDoubleSeconds(duration);
    double throughput = static_cast<double>(bytes_processed) / seconds;
    
    auto& metric = tensorstore::internal_metrics::Value<double>::New(
        metric_name,
        {{"unit", "bytes_per_second"},
         {"description", "HDF5 I/O throughput"}});
    
    metric.Set(throughput);
    
    std::cout << metric_name << ": "
              << throughput / (1024 * 1024) << " MB/s ("
              << seconds << "s)" << std::endl;
}

TEST(HDF5PerformanceTest, LargeDatasetAccess) {
    const size_t size = 1024 * 1024 * 1024;  // 1GB
    auto driver = CreateLargeTestDataset(size);
    ASSERT_NE(driver, nullptr);
    
    size_t dim_size = static_cast<size_t>(std::cbrt(size / sizeof(float)));
    size_t total_elements = dim_size * dim_size * dim_size;
    
    // Create test data
    std::vector<float> write_data(total_elements, 1.5f);
    std::vector<float> read_data(total_elements);
    
    // Test sequential write performance
    auto write_start = absl::Now();
    EXPECT_TRUE(driver->Write(write_data.data(), {0, 0, 0},
                             {dim_size, dim_size, dim_size}).ok());
    auto write_duration = absl::Now() - write_start;
    
    RecordBenchmarkResult("HDF5SequentialWrite", write_duration, size);
    
    // Test sequential read performance
    auto read_start = absl::Now();
    EXPECT_TRUE(driver->Read(read_data.data(), {0, 0, 0},
                            {dim_size, dim_size, dim_size}).ok());
    auto read_duration = absl::Now() - read_start;
    
    RecordBenchmarkResult("HDF5SequentialRead", read_duration, size);
    
    // Test random access performance
    const int num_random_ops = 1000;
    const size_t block_size = 16;  // 16^3 blocks
    
    // Random writes
    write_start = absl::Now();
    std::vector<float> small_write_data(block_size * block_size * block_size, 2.0f);
    
    for (int i = 0; i < num_random_ops; ++i) {
        size_t x = rand() % (dim_size - block_size);
        size_t y = rand() % (dim_size - block_size);
        size_t z = rand() % (dim_size - block_size);
        
        EXPECT_TRUE(driver->Write(small_write_data.data(),
                                {x, y, z},
                                {block_size, block_size, block_size}).ok());
    }
    write_duration = absl::Now() - write_start;
    
    RecordBenchmarkResult("HDF5RandomWrite", write_duration,
                         num_random_ops * block_size * block_size * block_size * sizeof(float));
    
    // Random reads
    read_start = absl::Now();
    std::vector<float> small_read_data(block_size * block_size * block_size);
    
    for (int i = 0; i < num_random_ops; ++i) {
        size_t x = rand() % (dim_size - block_size);
        size_t y = rand() % (dim_size - block_size);
        size_t z = rand() % (dim_size - block_size);
        
        EXPECT_TRUE(driver->Read(small_read_data.data(),
                               {x, y, z},
                               {block_size, block_size, block_size}).ok());
    }
    read_duration = absl::Now() - read_start;
    
    RecordBenchmarkResult("HDF5RandomRead", read_duration,
                         num_random_ops * block_size * block_size * block_size * sizeof(float));
}

}  // namespace
