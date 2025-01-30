#include "tensorstore/driver/hdf5/driver.h"

#include <iostream>
#include <vector>
#include <string>

#include "absl/status/status.h"
#include "tensorstore/context.h"
#include "tensorstore/index.h"
#include "tensorstore/spec.h"
#include "tensorstore/array.h"
#include "tensorstore/driver/registry.h"
#include "tensorstore/open.h"
#include "tensorstore/util/result.h"
#include "tensorstore/util/status.h"

using ::tensorstore::Context;
using ::tensorstore::Index;
using ::tensorstore::Spec;
using ::tensorstore::TensorStore;

namespace {

// Example 1: Basic Usage
void BasicExample() {
    std::cout << "\n=== Basic HDF5 Example ===\n";
    
    // Create a new dataset specification
    auto spec = Spec::FromJson({
        {"driver", "hdf5"},
        {"path", "example.h5"},
        {"dataset", "data"},
        {"dtype", "float32"},
        {"shape", {1000, 1000}},
        {"compression", {
            {"type", "gzip"},
            {"level", 6}
        }}
    }).value();
    
    // Open the dataset
    auto store = tensorstore::Open(spec).value();
    
    // Create sample data
    std::vector<float> write_data(1000 * 1000, 1.5f);
    
    // Write data
    auto write_future = store.Write(tensorstore::MakeArray(write_data));
    write_future.value();
    
    std::cout << "Successfully wrote data to dataset\n";
    
    // Read a subset of the data
    auto subset = store | tensorstore::Dims(0, 1).IndexSlice({0, 0}, {10, 10});
    auto read_result = subset.Read().value();
    
    std::cout << "First 10x10 elements:\n";
    for (Index i = 0; i < 10; ++i) {
        for (Index j = 0; j < 10; ++j) {
            std::cout << read_result(i, j) << " ";
        }
        std::cout << "\n";
    }
}

// Example 2: Advanced Features
void AdvancedExample() {
    std::cout << "\n=== Advanced HDF5 Example ===\n";
    
    // Create a dataset with chunking and compression
    auto spec = Spec::FromJson({
        {"driver", "hdf5"},
        {"path", "advanced_example.h5"},
        {"dataset", "data"},
        {"dtype", "float32"},
        {"shape", {100, 100, 100}},
        {"chunk_layout", {
            {"grid_origin", {0, 0, 0}},
            {"inner_order", {0, 1, 2}},
            {"chunk", {20, 20, 20}}
        }},
        {"compression", {
            {"type", "gzip"},
            {"level", 6}
        }}
    }).value();
    
    auto store = tensorstore::Open(spec).value();
    
    // Write data in chunks
    std::vector<float> chunk_data(20 * 20 * 20);
    for (size_t i = 0; i < chunk_data.size(); ++i) {
        chunk_data[i] = static_cast<float>(i);
    }
    
    // Write a single chunk
    auto chunk_store = store | tensorstore::Dims(0, 1, 2)
                                .IndexSlice({0, 0, 0}, {20, 20, 20});
    chunk_store.Write(tensorstore::MakeArray(chunk_data)).value();
    
    std::cout << "Successfully wrote chunk data\n";
    
    // Demonstrate attribute handling
    auto driver = store.driver();
    driver.SetAttribute("description", "Example HDF5 dataset").value();
    driver.SetAttribute("creation_time", 
                       std::string(absl::FormatTime("%Y-%m-%d %H:%M:%S", 
                                                  absl::Now(),
                                                  absl::LocalTimeZone()))).value();
    
    // Read attributes
    std::string description;
    driver.GetAttribute("description", &description).value();
    std::cout << "Dataset description: " << description << "\n";
    
    std::string creation_time;
    driver.GetAttribute("creation_time", &creation_time).value();
    std::cout << "Creation time: " << creation_time << "\n";
}

// Example 3: Error Handling
void ErrorHandlingExample() {
    std::cout << "\n=== Error Handling Example ===\n";
    
    // Try to open a non-existent file
    auto result = tensorstore::Open({
        {"driver", "hdf5"},
        {"path", "nonexistent.h5"},
        {"dataset", "data"}
    });
    
    if (!result.ok()) {
        std::cout << "Expected error opening non-existent file: "
                  << result.status() << "\n";
    }
    
    // Try to write invalid data
    auto spec = Spec::FromJson({
        {"driver", "hdf5"},
        {"path", "error_example.h5"},
        {"dataset", "data"},
        {"dtype", "float32"},
        {"shape", {10, 10}}
    }).value();
    
    auto store = tensorstore::Open(spec).value();
    
    // Try to write data with wrong dimensions
    std::vector<float> invalid_data(5 * 5, 1.0f);  // Wrong size
    auto write_result = store.Write(tensorstore::MakeArray(invalid_data));
    
    if (!write_result.ok()) {
        std::cout << "Expected error writing invalid data: "
                  << write_result.status() << "\n";
    }
}

}  // namespace

int main() {
    try {
        BasicExample();
        AdvancedExample();
        ErrorHandlingExample();
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}
