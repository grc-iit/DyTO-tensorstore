load("//tensorstore:tensorstore.bzl", "tensorstore_cc_library", "tensorstore_cc_test", "tensorstore_cc_binary")

package(default_visibility = ["//visibility:public"])

# Core HDF5 driver library
tensorstore_cc_library(
    name = "hdf5",
    srcs = [
        "src/driver.cc",
        "src/metadata.cc",
        "src/chunk_cache.cc",
        "src/schema.cc",
        "src/hdf5_util.cc",
        "src/attribute_manager.cc",
        "src/compound_type.cc",
        "src/error_handling.cc",
        "src/group.cc",
    ],
    hdrs = [
        "include/tensorstore/driver/hdf5/driver.h",
        "include/tensorstore/driver/hdf5/metadata.h",
        "include/tensorstore/driver/hdf5/chunk_cache.h",
        "include/tensorstore/driver/hdf5/schema.h",
        "include/tensorstore/driver/hdf5/hdf5_util.h",
        "include/tensorstore/driver/hdf5/attribute_manager.h",
        "include/tensorstore/driver/hdf5/compound_type.h",
        "include/tensorstore/driver/hdf5/error_handling.h",
        "include/tensorstore/driver/hdf5/group.h",
    ],
    deps = [
        "//tensorstore:context",
        "//tensorstore:data_type",
        "//tensorstore:index",
        "//tensorstore:rank",
        "//tensorstore/driver",
        "//tensorstore/internal:cache",
        "//tensorstore/internal:chunk_cache",
        "//tensorstore/util:result",
        "//tensorstore/util:str_cat",
        "@com_google_absl//absl/status",
        "@hdf5",
    ],
)

# Unit tests
tensorstore_cc_test(
    name = "unit_tests",
    srcs = [
        "tests/unit/driver_test.cc",
        "tests/unit/chunk_cache_test.cc",
        "tests/unit/hdf5_test.cc",
        "tests/unit/advanced_driver_test.cc",
    ],
    deps = [
        ":hdf5",
        "//tensorstore/util:status_testutil",
        "@com_google_googletest//:gtest_main",
    ],
)

# Integration tests
tensorstore_cc_test(
    name = "integration_tests",
    srcs = [
        "tests/integration/integration_test.cc",
    ],
    deps = [
        ":hdf5",
        "//tensorstore/util:status_testutil",
        "@com_google_googletest//:gtest_main",
    ],
)

# Performance tests
tensorstore_cc_test(
    name = "performance_tests",
    srcs = [
        "tests/performance/performance_test.cc",
    ],
    deps = [
        ":hdf5",
        "//tensorstore/util:status_testutil",
        "@com_google_googletest//:gtest_main",
    ],
)

# Stress tests
tensorstore_cc_test(
    name = "stress_tests",
    srcs = [
        "tests/stress/stress_test.cc",
    ],
    deps = [
        ":hdf5",
        "//tensorstore/util:status_testutil",
        "@com_google_googletest//:gtest_main",
    ],
)

# Examples
tensorstore_cc_binary(
    name = "examples/hdf5_example",
    srcs = ["examples/hdf5_example.cc"],
    deps = [":hdf5"],
)

tensorstore_cc_binary(
    name = "examples/hdf5_benchmark",
    srcs = ["examples/hdf5_benchmark.cc"],
    deps = [":hdf5"],
)
