#include "tensorstore/driver/hdf5/driver.h"

#include <memory>
#include <vector>
#include <string>
#include <thread>
#include <random>
#include <atomic>

#include "absl/status/status.h"
#include "absl/synchronization/mutex.h"
#include "absl/time/time.h"
#include "gtest/gtest.h"
#include "tensorstore/context.h"
#include "tensorstore/data_type.h"
#include "tensorstore/driver/hdf5/metadata.h"
#include "tensorstore/driver/hdf5/schema.h"
#include "tensorstore/index.h"
#include "tensorstore/internal/thread/thread_pool.h"
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

// Atomic counter for tracking concurrent operations
std::atomic<int64_t> g_successful_reads{0};
std::atomic<int64_t> g_successful_writes{0};
std::atomic<int64_t> g_failed_reads{0};
std::atomic<int64_t> g_failed_writes{0};

class HDF5StressTest : public ::testing::Test {
protected:
    void SetUp() override {
        temp_dir_ = std::make_unique<TempDir>();
        test_path_ = temp_dir_->path() / "stress_test.h5";
        
        // Create a test dataset
        Schema schema;
        schema.dtype(DataType::Float32())
              .rank(3)
              .shape({100, 100, 100})
              .chunk_layout({10, 10, 10});
        
        driver_ = std::make_unique<HDF5Driver>();
        TENSORSTORE_CHECK_OK(driver_->Initialize(schema));
        TENSORSTORE_CHECK_OK(driver_->SetFilePath(test_path_));
        
        // Initialize with some data
        std::vector<float> init_data(100 * 100 * 100, 1.0f);
        TENSORSTORE_CHECK_OK(driver_->Write(init_data.data(), {0, 0, 0}, {100, 100, 100}));
    }
    
    void TearDown() override {
        driver_.reset();
        temp_dir_.reset();
    }
    
    std::unique_ptr<TempDir> temp_dir_;
    std::string test_path_;
    std::unique_ptr<HDF5Driver> driver_;
};

void StressWorker(HDF5Driver* driver, int thread_id, int num_ops) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> op_dist(0, 1);  // 0 = read, 1 = write
    std::uniform_int_distribution<> pos_dist(0, 89);  // Leave room for block size
    
    const int block_size = 10;
    std::vector<float> write_buffer(block_size * block_size * block_size, 
                                  static_cast<float>(thread_id));
    std::vector<float> read_buffer(block_size * block_size * block_size);
    
    for (int i = 0; i < num_ops; ++i) {
        int x = pos_dist(gen);
        int y = pos_dist(gen);
        int z = pos_dist(gen);
        
        if (op_dist(gen) == 0) {
            // Read operation
            auto status = driver->Read(read_buffer.data(),
                                    {x, y, z},
                                    {block_size, block_size, block_size});
            if (status.ok()) {
                g_successful_reads++;
            } else {
                g_failed_reads++;
            }
        } else {
            // Write operation
            auto status = driver->Write(write_buffer.data(),
                                     {x, y, z},
                                     {block_size, block_size, block_size});
            if (status.ok()) {
                g_successful_writes++;
            } else {
                g_failed_writes++;
            }
        }
    }
}

TEST_F(HDF5StressTest, ConcurrentAccess) {
    const int num_threads = 10;
    const int ops_per_thread = 100;
    
    // Reset counters
    g_successful_reads = 0;
    g_successful_writes = 0;
    g_failed_reads = 0;
    g_failed_writes = 0;
    
    std::vector<std::thread> threads;
    for (int i = 0; i < num_threads; ++i) {
        threads.emplace_back(StressWorker, driver_.get(), i, ops_per_thread);
    }
    
    for (auto& thread : threads) {
        thread.join();
    }
    
    std::cout << "\nConcurrent Access Results:\n"
              << "Successful reads: " << g_successful_reads << "\n"
              << "Successful writes: " << g_successful_writes << "\n"
              << "Failed reads: " << g_failed_reads << "\n"
              << "Failed writes: " << g_failed_writes << "\n";
    
    // Verify data integrity
    std::vector<float> verify_data(100 * 100 * 100);
    EXPECT_TRUE(driver_->Read(verify_data.data(), {0, 0, 0}, {100, 100, 100}).ok());
}

class ScopedMemoryLimit {
public:
    explicit ScopedMemoryLimit(size_t limit_bytes) {
        // Set HDF5 cache size to limit_bytes
        H5Pset_cache(H5P_FILE_ACCESS_DEFAULT, 0, limit_bytes / 2,
                    limit_bytes, 0.5);
    }
    
    ~ScopedMemoryLimit() {
        // Reset to default cache size
        H5Pset_cache(H5P_FILE_ACCESS_DEFAULT, 0, 
                    32 * 1024 * 1024,  // 32MB default
                    64 * 1024 * 1024,  // 64MB max
                    0.75);
    }
};

TEST_F(HDF5StressTest, MemoryLimits) {
    const size_t memory_limit = 100 * 1024 * 1024;  // 100MB
    const size_t dataset_size = 512 * 1024 * 1024;  // 512MB
    
    {
        ScopedMemoryLimit limit(memory_limit);
        
        // Create a large dataset
        Schema schema;
        schema.dtype(DataType::Float32())
              .rank(2)
              .shape({8192, 16384})  // ~512MB
              .chunk_layout({256, 256});  // 256KB chunks
        
        auto large_driver = std::make_unique<HDF5Driver>();
        TENSORSTORE_CHECK_OK(large_driver->Initialize(schema));
        TENSORSTORE_CHECK_OK(large_driver->SetFilePath(temp_dir_->path() / "large_test.h5"));
        
        // Write data in chunks to stay within memory limits
        std::vector<float> chunk_data(256 * 256, 1.0f);
        for (int x = 0; x < 8192; x += 256) {
            for (int y = 0; y < 16384; y += 256) {
                EXPECT_TRUE(large_driver->Write(chunk_data.data(),
                                             {x, y},
                                             {256, 256}).ok());
            }
        }
        
        // Read data in chunks
        std::vector<float> verify_chunk(256 * 256);
        for (int x = 0; x < 8192; x += 256) {
            for (int y = 0; y < 16384; y += 256) {
                EXPECT_TRUE(large_driver->Read(verify_chunk.data(),
                                            {x, y},
                                            {256, 256}).ok());
                
                // Verify chunk data
                for (const float& val : verify_chunk) {
                    EXPECT_FLOAT_EQ(val, 1.0f);
                }
            }
        }
    }
}

}  // namespace
