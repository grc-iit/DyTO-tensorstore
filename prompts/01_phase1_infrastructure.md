# Phase 1: Infrastructure Setup and Basic Driver Implementation

## Task 1: Project Structure Setup
1. Create the HDF5 driver directory structure:
```bash
mkdir -p tensorstore/driver/hdf5
```

2. Create initial files:
```cpp
// driver.h
// metadata.h
// chunk_cache.h
// schema.h
// BUILD
```

3. Set up BUILD file with HDF5 dependencies:
```python
cc_library(
    name = "hdf5",
    srcs = [
        "driver.cc",
        "metadata.cc",
        "chunk_cache.cc",
        "schema.cc",
    ],
    hdrs = [
        "driver.h",
        "metadata.h",
        "chunk_cache.h",
        "schema.h",
    ],
    deps = [
        "//tensorstore:context",
        "//tensorstore:data_type",
        "//tensorstore:index",
        "//tensorstore:rank",
        "//tensorstore/driver",
        "//tensorstore/internal:cache",
        "//tensorstore/internal:chunk_cache",
        "@hdf5",
    ],
)
```

## Task 2: Basic Driver Class Implementation

1. Create the base HDF5Driver class:
```cpp
// driver.h
class HDF5Driver : public RegisteredDriver<HDF5Driver> {
public:
    static constexpr const char id[] = "hdf5";
    
    DataType dtype() override;
    DimensionIndex rank() override;
    
    Result<Schema> GetSchema() override;
    Result<ChunkLayout> GetChunkLayout() override;
    
    void Read(ReadRequest request, ReadChunkReceiver receiver) override;
    void Write(WriteRequest request, WriteChunkReceiver receiver) override;
    
private:
    std::shared_ptr<HDF5Cache> cache_;
    HDF5Metadata metadata_;
};
```

2. Implement basic driver registration:
```cpp
// driver.cc
namespace {
const internal::DriverRegistration<HDF5Driver> registration;
}  // namespace
```

## Task 3: HDF5 Integration Setup

1. Create HDF5 file handling utilities:
```cpp
// driver.cc
Result<hid_t> OpenHDF5File(const std::string& path, unsigned flags) {
    hid_t file_id = H5Fopen(path.c_str(), flags, H5P_DEFAULT);
    if (file_id < 0) {
        return absl::InternalError("Failed to open HDF5 file");
    }
    return file_id;
}

void CloseHDF5File(hid_t file_id) {
    if (file_id >= 0) {
        H5Fclose(file_id);
    }
}
```

2. Implement basic dataset operations:
```cpp
Result<hid_t> OpenDataset(hid_t file_id, const std::string& name) {
    hid_t dataset_id = H5Dopen2(file_id, name.c_str(), H5P_DEFAULT);
    if (dataset_id < 0) {
        return absl::InternalError("Failed to open dataset");
    }
    return dataset_id;
}
```

## Task 4: Metadata Structure Implementation

1. Create the metadata class:
```cpp
// metadata.h
struct HDF5Metadata {
    DimensionIndex rank;
    std::vector<Index> shape;
    std::vector<Index> chunks;
    DataType dtype;
    std::vector<std::string> dimension_labels;
    
    // HDF5 specific
    hid_t file_id;
    hid_t dataset_id;
    
    static Result<HDF5Metadata> Open(const std::string& path);
    Result<void> Close();
};
```

2. Implement metadata reading:
```cpp
// metadata.cc
Result<HDF5Metadata> HDF5Metadata::Open(const std::string& path) {
    HDF5Metadata metadata;
    TENSORSTORE_ASSIGN_OR_RETURN(metadata.file_id, OpenHDF5File(path, H5F_ACC_RDONLY));
    // Read dataset properties
    // Read dataspace
    // Read datatype
    return metadata;
}
```

## Task 5: Basic Schema Support

1. Create schema conversion utilities:
```cpp
// schema.h
Result<Schema> GetSchemaFromHDF5(const HDF5Metadata& metadata);
Result<void> ValidateSchema(const Schema& schema);
```

2. Implement basic schema conversion:
```cpp
// schema.cc
Result<Schema> GetSchemaFromHDF5(const HDF5Metadata& metadata) {
    Schema schema;
    schema.dtype(metadata.dtype);
    schema.rank(metadata.rank);
    // Set dimension properties
    return schema;
}
```

## Expected Outcomes

After completing Phase 1, you should have:
1. A basic driver structure that compiles
2. HDF5 file opening and closing functionality
3. Basic metadata reading capability
4. Initial schema conversion support

## Testing Tasks

1. Create basic unit tests:
```cpp
TEST(HDF5DriverTest, OpenFile) {
    auto result = HDF5Metadata::Open("test.h5");
    EXPECT_TRUE(result.ok());
}

TEST(HDF5DriverTest, ReadMetadata) {
    auto metadata = HDF5Metadata::Open("test.h5");
    ASSERT_TRUE(metadata.ok());
    EXPECT_GT(metadata->rank, 0);
}
```

## Next Steps

After completing these tasks, verify with the user and refine if necessary.
