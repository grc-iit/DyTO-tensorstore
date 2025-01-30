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
        ":driver",
    ],
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
        "//tensorstore:box",
        "//tensorstore:chunk_layout",
        "//tensorstore:codec_spec",
        "//tensorstore:contiguous_layout",
        "//tensorstore:data_type",
        "//tensorstore:index",
        "//tensorstore:json_serialization_options_base",
        "//tensorstore:rank",
        "//tensorstore:schema",
        "//tensorstore:strided_layout",
        "//tensorstore/index_space:dimension_units",
        "//tensorstore/index_space:index_transform",
        "//tensorstore/internal:json_metadata_matching",
        "//tensorstore/internal:type_traits",
        "//tensorstore/internal/json:same",
        "//tensorstore/internal/json_binding",
        "//tensorstore/internal/json_binding:bindable",
        "//tensorstore/internal/json_binding:data_type",
        "//tensorstore/internal/json_binding:dimension_indexed",
        "//tensorstore/internal/riegeli:array_endian_codec",
        "//tensorstore/serialization",
        "//tensorstore/serialization:json",
        "//tensorstore/util:constant_vector",
        "//tensorstore/util:endian",
        "//tensorstore/util:extents",
        "//tensorstore/util:result",
        "//tensorstore/util:span",
        "//tensorstore/util:status",
        "//tensorstore/util:str_cat",
        "//tensorstore/util:unit",
        "//tensorstore/util/garbage_collection",
        "@com_github_nlohmann_json//:json",
        "@com_google_absl//absl/algorithm:container",
        "@com_google_absl//absl/meta:type_traits",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:cord",
        "@com_google_riegeli//riegeli/bytes:cord_reader",
        "@com_google_riegeli//riegeli/bytes:cord_writer",
        "@com_google_riegeli//riegeli/bytes:reader",
        "@com_google_riegeli//riegeli/bytes:writer",
        "@com_google_riegeli//riegeli/endian:endian_reading",
        "@com_google_riegeli//riegeli/endian:endian_writing",
    ],
    alwayslink = 1,
)

tensorstore_cc_library(
    name = "driver",
    srcs = ["driver.cc"],
    deps = [
        ":metadata",
        "//tensorstore:array",
        "//tensorstore:array_storage_statistics",
        "//tensorstore:box",
        "//tensorstore:chunk_layout",
        "//tensorstore:codec_spec",
        "//tensorstore:contiguous_layout",
        "//tensorstore:data_type",
        "//tensorstore:index",
        "//tensorstore:index_interval",
        "//tensorstore:open_mode",
        "//tensorstore:open_options",
        "//tensorstore:rank",
        "//tensorstore:transaction",
        "//tensorstore/driver",
        "//tensorstore/driver:chunk",
        "//tensorstore/driver:kvs_backed_chunk_driver",
        "//tensorstore/index_space:dimension_units",
        "//tensorstore/index_space:index_transform",
        "//tensorstore/internal:async_write_array",
        "//tensorstore/internal:chunk_grid_specification",
        "//tensorstore/internal:grid_chunk_key_ranges_base10",
        "//tensorstore/internal:intrusive_ptr",
        "//tensorstore/internal:lexicographical_grid_index_key",
        "//tensorstore/internal:storage_statistics",
        "//tensorstore/internal/cache",
        "//tensorstore/internal/cache_key",
        "//tensorstore/internal/json_binding",
        "//tensorstore/util:dimension_set",
        "//tensorstore/util:executor",
        "//tensorstore/util:future",
        "//tensorstore/util:result",
        "//tensorstore/util:span",
        "//tensorstore/util:status",
        "//tensorstore/util:str_cat",
        "//tensorstore/util/execution:any_receiver",
        "//tensorstore/util/garbage_collection",
        "@com_github_nlohmann_json//:json",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings",
        "@com_google_absl//absl/strings:cord",
        "@com_google_absl//absl/time",
    ],
    alwayslink = 1,
)

