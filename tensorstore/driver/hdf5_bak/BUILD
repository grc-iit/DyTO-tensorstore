load("//bazel:tensorstore.bzl", "tensorstore_cc_library", "tensorstore_cc_test")
load("//docs:doctest.bzl", "doctest_test")

package(default_visibility = ["//visibility:public"])

licenses(["notice"])

DOCTEST_SOURCES = glob([
    "**/*.rst",
    "**/*.yml",
])

doctest_test(
    name = "doctest_test",
    srcs = DOCTEST_SOURCES,
)

filegroup(
    name = "doc_sources",
    srcs = DOCTEST_SOURCES,
)

tensorstore_cc_library(
    name = "hdf5",
    deps = [
        ":gzip_compressor",
        ":driver",
    ],
)

tensorstore_cc_library(
    name = "gzip_compressor",
    srcs = ["gzip_compressor.cc"],
    deps = [
        ":compressor",
        "//tensorstore/internal/compression:zlib_compressor",
        "//tensorstore/internal/json_binding",
    ],
    alwayslink = 1,
)

tensorstore_cc_library(
    name = "compressor",
    srcs = ["compressor.cc"],
    hdrs = [
        "compressor.h",
        "compressor_registry.h",
    ],
    deps = [
        "//tensorstore/internal:json_registry",
        "//tensorstore/internal/compression:json_specified_compressor",
        "//tensorstore/internal/json_binding",
        "//tensorstore/internal/json_binding:bindable",
        "@com_google_absl//absl/base:no_destructor",
    ],
)

tensorstore_cc_library(
    name = "metadata",
    srcs = ["metadata.cc"],
    hdrs = ["metadata.h"],
    deps = [
        ":compressor",
        "//tensorstore:array",
        "//tensorstore:contiguous_layout",
        "//tensorstore:data_type",
        "//tensorstore:index",
        "//tensorstore:json_serialization_options_base",
        "//tensorstore:rank",
        "//tensorstore:strided_layout",
        "//tensorstore/internal:data_type_endian_conversion",
        "//tensorstore/internal:flat_cord_builder",
        "//tensorstore/internal/json",
        "//tensorstore/internal/json:value_as",
        "//tensorstore/internal/json_binding",
        "//tensorstore/internal/json_binding:bindable",
        "//tensorstore/internal/json_binding:data_type",
        "//tensorstore/internal/json_binding:dimension_indexed",
        "//tensorstore/internal/riegeli:array_endian_codec",
        "//tensorstore/serialization",
        "//tensorstore/serialization:json",
        "//tensorstore/util:byte_strided_pointer",
        "//tensorstore/util:element_pointer",
        "//tensorstore/util:endian",
        "//tensorstore/util:extents",
        "//tensorstore/util:result",
        "//tensorstore/util:span",
        "//tensorstore/util:status",
        "//tensorstore/util:str_cat",
        "//tensorstore/util/garbage_collection",
        "@com_github_nlohmann_json//:json",
        "@com_google_absl//absl/base:core_headers",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/meta:type_traits",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:cord",
        "@com_google_riegeli//riegeli/bytes:cord_reader",
        "@com_google_riegeli//riegeli/bytes:cord_writer",
        "@com_google_riegeli//riegeli/bytes:read_all",
        "@com_google_riegeli//riegeli/bytes:reader",
        "@com_google_riegeli//riegeli/bytes:write",
        "@com_google_riegeli//riegeli/bytes:writer",
    ],
)

tensorstore_cc_library(
    name = "driver",
    srcs = ["driver.cc"],
    hdrs = ["driver_impl.h"],
    deps = [
        ":metadata",
        "//tensorstore",
        "//tensorstore:array",
        "//tensorstore:array_storage_statistics",
        "//tensorstore:box",
        "//tensorstore:chunk_layout",
        "//tensorstore:codec_spec",
        "//tensorstore:context",
        "//tensorstore:contiguous_layout",
        "//tensorstore:data_type",
        "//tensorstore:index",
        "//tensorstore:index_interval",
        "//tensorstore:open_mode",
        "//tensorstore:open_options",
        "//tensorstore:rank",
        "//tensorstore:transaction",
        "//tensorstore/driver",
        "//tensorstore/driver:kvs_backed_chunk_driver",
        "//tensorstore/index_space:index_transform",
        "//tensorstore/index_space:transform_broadcastable_array",
        "//tensorstore/internal:async_write_array",
        "//tensorstore/internal:chunk_grid_specification",
        "//tensorstore/internal:grid_storage_statistics",
        "//tensorstore/internal:type_traits",
        "//tensorstore/internal/cache",
        "//tensorstore/internal/cache:chunk_cache",
        "//tensorstore/internal/cache_key",
        "//tensorstore/internal/json_binding",
        "//tensorstore/internal/json_binding:bindable",
        "//tensorstore/kvstore",
        "//tensorstore/util:dimension_set",
        "//tensorstore/util:executor",
        "//tensorstore/util:future",
        "//tensorstore/util:result",
        "//tensorstore/util:span",
        "//tensorstore/util:status",
        "//tensorstore/util:str_cat",
        "//tensorstore/util/garbage_collection",
        "@com_google_absl//absl/container:inlined_vector",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings:cord",
    ],
    alwayslink = 1,
)

tensorstore_cc_test(
    name = "storage_statistics_test",
    size = "small",
    srcs = ["storage_statistics_test.cc"],
    deps = [
        ":driver",
        "//tensorstore/driver/zarr:storage_statistics_test_util",
        "@com_google_googletest//:gtest_main",
    ],
)
