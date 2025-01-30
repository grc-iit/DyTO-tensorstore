# Phase 5: Testing and Documentation

## Task 1: Unit Testing Framework

1. Set up test infrastructure:
```cpp
// driver_test.h
class HDF5DriverTest : public ::testing::Test {
protected:
    void SetUp() override {
        temp_dir_ = absl::make_unique<TestTempDir>();
        test_path_ = temp_dir_->path / "test.h5";
    }
    
    void TearDown() override {
        temp_dir_.reset();
    }
    
    std::unique_ptr<TestTempDir> temp_dir_;
    std::string test_path_;
};

// Create test utilities
Result<DriverHandle> CreateTestDriver(
    const TestTempDir& temp_dir,
    const Schema& schema,
    const CompressionParams& compression = {}) {
    // Implementation
}
```

2. Implement basic test cases:
```cpp
// driver_test.cc
TEST_F(HDF5DriverTest, CreateAndOpen) {
    Schema schema;
    schema.dtype(DataType::Float32())
          .rank(2)
          .shape({100, 200});
    
    TENSORSTORE_ASSERT_OK_AND_ASSIGN(
        auto driver,
        CreateTestDriver(*temp_dir_, schema));
    
    EXPECT_EQ(driver->dtype(), DataType::Float32());
    EXPECT_EQ(driver->rank(), 2);
}

TEST_F(HDF5DriverTest, ReadWrite) {
    auto driver = CreateTestDriver(/*...*/);
    
    // Test data
    std::vector<float> data(1000, 1.5f);
    
    // Write
    EXPECT_TRUE(driver->Write(data).ok());
    
    // Read and verify
    auto read_result = driver->Read();
    EXPECT_TRUE(read_result.ok());
    EXPECT_EQ(read_result->data, data);
}
```

## Task 2: Integration Testing

1. Implement end-to-end tests:
```cpp
// integration_test.cc
TEST(HDF5IntegrationTest, CompleteWorkflow) {
    // 1. Create dataset
    auto driver = CreateTestDataset();
    
    // 2. Write data with compression
    WriteTestData(driver);
    
    // 3. Read and verify
    VerifyTestData(driver);
    
    // 4. Modify attributes
    ModifyAndVerifyAttributes(driver);
    
    // 5. Test group operations
    TestGroupOperations(driver);
}
```

2. Implement performance tests:
```cpp
// performance_test.cc
TEST(HDF5PerformanceTest, LargeDatasetAccess) {
    const size_t size = 1024 * 1024 * 1024;  // 1GB
    auto driver = CreateLargeTestDataset(size);
    
    // Measure write performance
    auto start = absl::Now();
    EXPECT_TRUE(driver->Write(large_data).ok());
    auto write_duration = absl::Now() - start;
    
    // Measure read performance
    start = absl::Now();
    auto read_result = driver->Read();
    auto read_duration = absl::Now() - start;
    
    RecordBenchmarkResult("HDF5Write", write_duration);
    RecordBenchmarkResult("HDF5Read", read_duration);
}
```

## Task 3: Stress Testing

1. Implement concurrent access tests:
```cpp
// stress_test.cc
TEST(HDF5StressTest, ConcurrentAccess) {
    auto driver = CreateTestDriver();
    
    std::vector<std::thread> threads;
    for (int i = 0; i < 10; ++i) {
        threads.emplace_back([driver, i]() {
            for (int j = 0; j < 100; ++j) {
                // Randomly read or write
                if (rand() % 2) {
                    driver->Read(/*...*/);
                } else {
                    driver->Write(/*...*/);
                }
            }
        });
    }
    
    for (auto& thread : threads) {
        thread.join();
    }
}
```

2. Implement resource limit tests:
```cpp
// stress_test.cc
TEST(HDF5StressTest, MemoryLimits) {
    // Test with limited memory
    ScopedMemoryLimit memory_limit(100 * 1024 * 1024);  // 100MB
    
    auto driver = CreateLargeTestDataset(1024 * 1024 * 1024);  // 1GB
    EXPECT_TRUE(driver->Read().ok());  // Should work within memory limit
}
```

## Task 4: Documentation

1. Create driver documentation:
```markdown
# HDF5 Driver for TensorStore

## Overview
The HDF5 driver provides access to HDF5 format arrays through the TensorStore interface.
It supports reading, writing, and creating new arrays with various data types and
compression options.

## Features
- Full HDF5 format support
- Chunked access
- Compression (GZIP, SZIP)
- Attribute handling
- Group support
- Compound data types

## Usage Example
```cpp
auto spec = tensorstore::Spec::FromJson({
    {"driver", "hdf5"},
    {"path", "path/to/file.h5"},
    {"dataset", "data"},
    {"dtype", "float32"},
    {"compression", {"type": "gzip", "level": 6}}
});

TENSORSTORE_ASSIGN_OR_RETURN(auto store, tensorstore::Open(spec).result());
```
```

2. Create API documentation:
```cpp
/// @file driver.h
/// @brief HDF5 driver implementation for TensorStore.

/// @class HDF5Driver
/// @brief Provides access to HDF5 datasets through TensorStore interface.
///
/// This driver implements the TensorStore interface for HDF5 files, supporting
/// both reading and writing operations with optional compression and chunking.
///
/// @see tensorstore::Driver

/// @function ReadChunk
/// @brief Reads a chunk of data from the HDF5 dataset.
/// @param indices The chunk indices to read.
/// @return A Future that resolves to the chunk data.
```

## Task 5: Example Code

1. Create usage examples:
```cpp
// examples/hdf5_example.cc
#include "tensorstore/driver/hdf5/driver.h"

int main() {
    // 1. Create a new dataset
    auto spec = tensorstore::Spec::FromJson({
        {"driver", "hdf5"},
        {"path", "example.h5"},
        {"dataset", "data"},
        {"dtype", "float32"},
        {"shape", {1000, 1000}},
        {"compression", {"type": "gzip", "level": 6}}
    });
    
    auto store = tensorstore::Open(spec).value();
    
    // 2. Write data
    std::vector<float> data(1000 * 1000, 1.5f);
    store.Write(data).value();
    
    // 3. Read subset
    auto subset = store | tensorstore::Dims(0, 1).IndexSlice({0, 0}, {100, 100});
    auto result = subset.Read().value();
    
    return 0;
}
```

2. Create benchmark examples:
```cpp
// examples/hdf5_benchmark.cc
void BenchmarkLargeRead() {
    auto store = OpenTestDataset(1024 * 1024 * 1024);  // 1GB
    
    auto start = absl::Now();
    auto result = store.Read().value();
    auto duration = absl::Now() - start;
    
    std::cout << "Read 1GB in: " << duration << std::endl;
}
```

## Expected Outcomes

After completing Phase 5, you should have:
1. Comprehensive test suite
2. Performance benchmarks
3. Complete documentation
4. Usage examples

## Final Checklist

1. Tests:
- [ ] Unit tests for all components
- [ ] Integration tests for workflows
- [ ] Performance benchmarks
- [ ] Stress tests
- [ ] Memory leak tests

2. Documentation:
- [ ] API documentation
- [ ] Usage guide
- [ ] Examples
- [ ] Performance guidelines

3. Code Quality:
- [ ] Code formatting
- [ ] Error handling
- [ ] Comments and docstrings
- [ ] Memory management

## Next Steps

1. Submit for code review
2. Address feedback
3. Prepare for integration into main codebase
