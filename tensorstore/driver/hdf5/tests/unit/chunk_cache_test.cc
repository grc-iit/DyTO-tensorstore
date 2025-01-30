#include <gtest/gtest.h>
#include <gmock/gmock.h>

#include "tensorstore/driver/hdf5/chunk_cache.h"
#include "tensorstore/driver/hdf5/metadata.h"
#include "tensorstore/util/status_testutil.h"
#include "tensorstore/util/result.h"

namespace {
namespace ts = tensorstore;
using ::testing::ElementsAreArray;
using ::testing::Eq;
using ::testing::IsTrue;

class HDF5CacheTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Create test dataset and metadata
        metadata_.dtype = ts::dtype_v<uint8_t>;
        metadata_.chunk_layout = {2, 2};  // 2x2 chunks
        metadata_.shape = {4, 4};         // 4x4 dataset
        
        // Create test data
        test_data_.resize(4);  // 2x2 chunk = 4 bytes
        std::iota(test_data_.begin(), test_data_.end(), 0);  // Fill with 0,1,2,3
        
        // Create large test data for eviction tests
        large_data_.resize(256);  // 256 bytes per chunk
        std::fill(large_data_.begin(), large_data_.end(), 0xFF);
        
        // Initialize cache with test dataset
        cache_ = std::make_unique<ts::hdf5_driver::HDF5ChunkCache>(
            /*dataset_id=*/-1,  // Mock dataset ID
            metadata_
        );
    }

    ts::hdf5_driver::HDF5Metadata metadata_;
    std::vector<unsigned char> test_data_;
    std::vector<unsigned char> large_data_;
    std::unique_ptr<ts::hdf5_driver::HDF5ChunkCache> cache_;
};

TEST_F(HDF5CacheTest, ReadWriteChunk) {
    ts::hdf5_driver::ChunkKey key{{0, 0}};
    
    // Write chunk
    TENSORSTORE_EXPECT_OK(cache_->WriteChunk(key, ts::span<const unsigned char>(test_data_)));
    
    // Read chunk
    TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto read_result, cache_->ReadChunk(key));
    EXPECT_THAT(read_result, ElementsAreArray(test_data_));
    
    // Verify cache statistics
    auto stats = cache_->GetStats();
    EXPECT_EQ(stats.num_entries, 1);
    EXPECT_EQ(stats.total_size, test_data_.size());
    EXPECT_EQ(stats.num_hits, 1);  // From the read operation
    EXPECT_EQ(stats.num_misses, 0);
}

TEST_F(HDF5CacheTest, Eviction) {
    const size_t max_size = 1024;  // 1KB cache size
    
    // Fill cache beyond capacity
    for (int i = 0; i < 10; ++i) {
        ts::hdf5_driver::ChunkKey key{{i, 0}};
        TENSORSTORE_EXPECT_OK(cache_->WriteChunk(key, ts::span<const unsigned char>(large_data_)));
    }
    
    // Force eviction
    cache_->EvictEntries(max_size);
    
    // Verify cache size is within limits
    auto stats = cache_->GetStats();
    EXPECT_LE(stats.total_size, max_size);
}

TEST_F(HDF5CacheTest, ParallelRead) {
    // Write multiple chunks
    std::vector<ts::hdf5_driver::ChunkKey> keys = {
        {{0, 0}},
        {{0, 1}},
        {{1, 0}},
        {{1, 1}}
    };
    
    for (const auto& key : keys) {
        TENSORSTORE_EXPECT_OK(cache_->WriteChunk(key, ts::span<const unsigned char>(test_data_)));
    }
    
    // Read chunks in parallel
    auto future = cache_->ReadMultipleChunks(ts::span<const ts::hdf5_driver::ChunkKey>(keys));
    TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto results, future.result());
    
    // Verify results
    EXPECT_EQ(results.size(), keys.size());
    for (const auto& result : results) {
        EXPECT_THAT(result, ElementsAreArray(test_data_));
    }
}

TEST_F(HDF5CacheTest, Prefetch) {
    ts::hdf5_driver::ChunkKey key{{0, 0}};
    
    // Prefetch chunk
    cache_->Prefetch(ts::span<const ts::hdf5_driver::ChunkKey>(&key, 1));
    
    // Wait a bit for prefetch to complete (in real code, would use proper synchronization)
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    
    // Verify chunk is in cache
    auto stats = cache_->GetStats();
    EXPECT_EQ(stats.num_entries, 1);
    EXPECT_EQ(stats.num_hits, 0);  // No explicit reads yet
    
    // Read the prefetched chunk
    TENSORSTORE_ASSERT_OK_AND_ASSIGN(auto read_result, cache_->ReadChunk(key));
    EXPECT_THAT(read_result, ElementsAreArray(test_data_));
    
    // Verify hit count increased
    stats = cache_->GetStats();
    EXPECT_EQ(stats.num_hits, 1);
}

}  // namespace
