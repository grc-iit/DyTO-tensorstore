#include "tensorstore/driver/hdf5/driver.h"

#include <iostream>
#include <vector>
#include <string>
#include <chrono>
#include <random>

#include "absl/status/status.h"
#include "absl/time/time.h"
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

// Helper function to format bytes in human-readable format
std::string FormatBytes(size_t bytes) {
    const char* units[] = {"B", "KB", "MB", "GB", "TB"};
    int unit = 0;
    double size = static_cast<double>(bytes);
    
    while (size >= 1024.0 && unit < 4) {
        size /= 1024.0;
        unit++;
    }
    
    char buffer[32];
    snprintf(buffer, sizeof(buffer), "%.2f %s", size, units[unit]);
    return std::string(buffer);
}

// Helper function to create a test dataset
TensorStore<> CreateTestDataset(size_t size_bytes, const std::string& path) {
    // Calculate dimensions for a square 2D dataset
    size_t elements = size_bytes / sizeof(float);
    size_t dim_size = static_cast<size_t>(std::sqrt(elements));
    
    auto spec = Spec::FromJson({
        {"driver", "hdf5"},
        {"path", path},
        {"dataset", "benchmark_data"},
        {"dtype", "float32"},
        {"shape", {dim_size, dim_size}},
        {"chunk_layout", {
            {"grid_origin", {0, 0}},
            {"inner_order", {0, 1}},
            {"chunk", {256, 256}}  // 256KB chunks
        }},
        {"compression", {
            {"type", "gzip"},
            {"level", 1}  // Light compression for better performance
        }}
    }).value();
    
    return tensorstore::Open(spec).value();
}

// Benchmark 1: Sequential Read/Write Performance
void BenchmarkSequentialAccess() {
    std::cout << "\n=== Sequential Access Benchmark ===\n";
    
    const size_t size = 1024 * 1024 * 1024;  // 1GB
    auto store = CreateTestDataset(size, "sequential_benchmark.h5");
    
    size_t dim_size = static_cast<size_t>(std::sqrt(size / sizeof(float)));
    std::vector<float> data(dim_size * dim_size, 1.5f);
    
    // Sequential Write
    auto write_start = absl::Now();
    store.Write(tensorstore::MakeArray(data)).value();
    auto write_duration = absl::Now() - write_start;
    
    double write_throughput = size / absl::ToDoubleSeconds(write_duration);
    std::cout << "Sequential Write:\n"
              << "  Size: " << FormatBytes(size) << "\n"
              << "  Time: " << write_duration << "\n"
              << "  Throughput: " << FormatBytes(static_cast<size_t>(write_throughput)) << "/s\n";
    
    // Sequential Read
    auto read_start = absl::Now();
    auto read_result = store.Read().value();
    auto read_duration = absl::Now() - read_start;
    
    double read_throughput = size / absl::ToDoubleSeconds(read_duration);
    std::cout << "Sequential Read:\n"
              << "  Size: " << FormatBytes(size) << "\n"
              << "  Time: " << read_duration << "\n"
              << "  Throughput: " << FormatBytes(static_cast<size_t>(read_throughput)) << "/s\n";
}

// Benchmark 2: Random Access Performance
void BenchmarkRandomAccess() {
    std::cout << "\n=== Random Access Benchmark ===\n";
    
    const size_t size = 1024 * 1024 * 1024;  // 1GB
    auto store = CreateTestDataset(size, "random_benchmark.h5");
    
    size_t dim_size = static_cast<size_t>(std::sqrt(size / sizeof(float)));
    const size_t block_size = 64;  // 64x64 blocks
    const int num_operations = 1000;
    
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dist(0, dim_size - block_size);
    
    // Random Write
    std::vector<float> write_data(block_size * block_size, 1.5f);
    auto write_start = absl::Now();
    
    for (int i = 0; i < num_operations; ++i) {
        Index x = dist(gen);
        Index y = dist(gen);
        
        auto block_store = store | tensorstore::Dims(0, 1)
                                    .IndexSlice({x, y}, {x + block_size, y + block_size});
        block_store.Write(tensorstore::MakeArray(write_data)).value();
    }
    
    auto write_duration = absl::Now() - write_start;
    size_t write_bytes = num_operations * block_size * block_size * sizeof(float);
    double write_throughput = write_bytes / absl::ToDoubleSeconds(write_duration);
    
    std::cout << "Random Write:\n"
              << "  Operations: " << num_operations << "\n"
              << "  Block Size: " << FormatBytes(block_size * block_size * sizeof(float)) << "\n"
              << "  Total Size: " << FormatBytes(write_bytes) << "\n"
              << "  Time: " << write_duration << "\n"
              << "  Throughput: " << FormatBytes(static_cast<size_t>(write_throughput)) << "/s\n";
    
    // Random Read
    std::vector<float> read_data(block_size * block_size);
    auto read_start = absl::Now();
    
    for (int i = 0; i < num_operations; ++i) {
        Index x = dist(gen);
        Index y = dist(gen);
        
        auto block_store = store | tensorstore::Dims(0, 1)
                                    .IndexSlice({x, y}, {x + block_size, y + block_size});
        auto result = block_store.Read().value();
    }
    
    auto read_duration = absl::Now() - read_start;
    size_t read_bytes = num_operations * block_size * block_size * sizeof(float);
    double read_throughput = read_bytes / absl::ToDoubleSeconds(read_duration);
    
    std::cout << "Random Read:\n"
              << "  Operations: " << num_operations << "\n"
              << "  Block Size: " << FormatBytes(block_size * block_size * sizeof(float)) << "\n"
              << "  Total Size: " << FormatBytes(read_bytes) << "\n"
              << "  Time: " << read_duration << "\n"
              << "  Throughput: " << FormatBytes(static_cast<size_t>(read_throughput)) << "/s\n";
}

// Benchmark 3: Compression Performance
void BenchmarkCompression() {
    std::cout << "\n=== Compression Benchmark ===\n";
    
    const size_t size = 256 * 1024 * 1024;  // 256MB
    std::vector<std::pair<std::string, int>> compression_configs = {
        {"none", 0},
        {"gzip", 1},
        {"gzip", 6},
        {"gzip", 9}
    };
    
    for (const auto& config : compression_configs) {
        std::string name = config.first;
        int level = config.second;
        
        auto spec = Spec::FromJson({
            {"driver", "hdf5"},
            {"path", "compression_benchmark_" + name + "_" + std::to_string(level) + ".h5"},
            {"dataset", "data"},
            {"dtype", "float32"},
            {"shape", {512, 512}},
            {"chunk_layout", {
                {"grid_origin", {0, 0}},
                {"inner_order", {0, 1}},
                {"chunk", {64, 64}}
            }},
            {"compression", {
                {"type", name},
                {"level", level}
            }}
        }).value();
        
        auto store = tensorstore::Open(spec).value();
        
        // Create test data with some patterns for better compression
        std::vector<float> data(512 * 512);
        for (size_t i = 0; i < data.size(); ++i) {
            data[i] = static_cast<float>(i % 1000);  // Repeating pattern
        }
        
        // Write benchmark
        auto write_start = absl::Now();
        store.Write(tensorstore::MakeArray(data)).value();
        auto write_duration = absl::Now() - write_start;
        
        // Read benchmark
        auto read_start = absl::Now();
        auto read_result = store.Read().value();
        auto read_duration = absl::Now() - read_start;
        
        std::cout << "\nCompression: " << name 
                  << (level > 0 ? " (level " + std::to_string(level) + ")" : "") << "\n"
                  << "  Write Time: " << write_duration << "\n"
                  << "  Read Time: " << read_duration << "\n";
    }
}

}  // namespace

int main() {
    try {
        BenchmarkSequentialAccess();
        BenchmarkRandomAccess();
        BenchmarkCompression();
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}
