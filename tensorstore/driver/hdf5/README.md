# TensorStore HDF5 Driver

The HDF5 driver for TensorStore provides high-performance, scalable access to HDF5 datasets with support for chunking, compression, and concurrent access. This document covers building, testing, and using the HDF5 driver.

## Table of Contents
- [Prerequisites](#prerequisites)
- [Building](#building)
- [Testing](#testing)
- [Examples](#examples)
- [Performance Tuning](#performance-tuning)
- [Troubleshooting](#troubleshooting)

## Prerequisites

### System Requirements
- C++17 compatible compiler (GCC 7+, Clang 7+, or MSVC 2019+)
- CMake 3.24+
- Bazel 5.0+
- Python 3.7+ (for Python bindings)

### Required Libraries
1. HDF5 Library (1.10.0+):
```bash
# Ubuntu/Debian
sudo apt-get install libhdf5-dev

# CentOS/RHEL
sudo yum install hdf5-devel

# macOS
brew install hdf5
```

2. Other Dependencies:
```bash
# Ubuntu/Debian
sudo apt-get install \
    build-essential \
    cmake \
    python3-dev \
    python3-pip \
    ninja-build

# Install Python dependencies
pip install -r requirements.txt
```

## Building

### Building with Bazel (Recommended)

1. Clone the repository:
```bash
git clone https://github.com/google/tensorstore.git
cd tensorstore
```

2. Build the HDF5 driver:
```bash
bazel build //tensorstore/driver/hdf5/...
```

3. Build Python bindings (optional):
```bash
bazel build //python/tensorstore:_tensorstore
```

### Building with CMake

1. Configure the build:
```bash
mkdir build && cd build
cmake .. -GNinja \
    -DTENSORSTORE_ENABLE_HDF5=ON \
    -DTENSORSTORE_ENABLE_PYTHON=ON
```

2. Build:
```bash
ninja
```

## Testing

### Running Tests

1. Unit Tests:
```bash
bazel test //tensorstore/driver/hdf5:unit_tests
```

2. Integration Tests:
```bash
bazel test //tensorstore/driver/hdf5:integration_tests
```

3. Performance Tests:
```bash
bazel run //tensorstore/driver/hdf5:performance_tests
```

4. Stress Tests:
```bash
bazel run //tensorstore/driver/hdf5:stress_tests
```

### Test Configuration

The test suite can be configured through environment variables:
```bash
# Set HDF5 cache size for tests
export HDF5_CACHE_SIZE=128M

# Enable verbose test output
export TENSORSTORE_TEST_VERBOSE=1

# Run tests with specific compression
export HDF5_TEST_COMPRESSION=gzip
```

## Examples

### Running Examples

1. Basic Usage Example:
```bash
bazel run //tensorstore/driver/hdf5/examples:hdf5_example
```

2. Benchmark Example:
```bash
bazel run //tensorstore/driver/hdf5/examples:hdf5_benchmark
```

### Example Code

Basic dataset creation and access:
```cpp
auto spec = tensorstore::Spec::FromJson({
    {"driver", "hdf5"},
    {"path", "example.h5"},
    {"dataset", "data"},
    {"dtype", "float32"},
    {"shape", {1000, 1000}},
    {"compression", {"type": "gzip", "level": 6}}
});

auto store = tensorstore::Open(spec).value();
```

## Performance Tuning

### Cache Configuration
```cpp
// Set chunk cache size
auto context = Context::Default();
context.SetCachePool(CachePool::Make(CachePool::Limits{
    .total_bytes_limit = 1024 * 1024 * 1024  // 1GB
}));
```

### Compression Settings
```cpp
// Configure compression
auto spec = tensorstore::Spec::FromJson({
    "compression": {
        "type": "gzip",
        "level": 6  // Balance between speed and compression
    }
});
```

### Chunking Optimization
```cpp
// Optimize chunk size for your access pattern
auto spec = tensorstore::Spec::FromJson({
    "chunk_layout": {
        "grid_origin": [0, 0],
        "inner_order": [0, 1],
        "chunk": [256, 256]  // Adjust based on your needs
    }
});
```

## Troubleshooting

### Common Issues

1. **Build Failures**
   - Ensure all dependencies are installed
   - Check compiler version compatibility
   - Verify HDF5 library installation

2. **Performance Issues**
   - Adjust chunk sizes to match access patterns
   - Configure appropriate cache sizes
   - Use compression judiciously

3. **Memory Issues**
   - Monitor HDF5 cache usage
   - Adjust chunk sizes if needed
   - Consider using memory-mapped I/O

### Getting Help

1. Check the [TensorStore documentation](https://google.github.io/tensorstore)
2. File issues on GitHub
3. Join the TensorStore discussion forum

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request

Please follow the [coding style guide](https://google.github.io/tensorstore/development.html) and ensure all tests pass.
