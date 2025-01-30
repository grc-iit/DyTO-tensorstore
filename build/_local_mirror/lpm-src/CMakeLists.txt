set(CMAKE_MESSAGE_INDENT "[lpm] ")

execute_process(
  COMMAND ${CMAKE_COMMAND} -E copy_if_different
       "/home/yejie/SCS_lab/DyTO/DyTO/tensorstore/third_party/local_proto_mirror/src/lpm.BUILD.bazel"
       "${CMAKE_CURRENT_SOURCE_DIR}/BUILD.bazel"
  WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
)

execute_process(
  COMMAND ${CMAKE_COMMAND} -E copy_if_different
       "/home/yejie/SCS_lab/DyTO/DyTO/tensorstore/third_party/local_proto_mirror/src/imports.bzl"
       "${CMAKE_CURRENT_SOURCE_DIR}/imports.bzl"
  WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
)

execute_process(
  COMMAND ${CMAKE_COMMAND} -E copy_if_different
       "/home/yejie/SCS_lab/DyTO/DyTO/tensorstore/third_party/local_proto_mirror/src/lpm.BUILD.bazel"
       "${CMAKE_CURRENT_SOURCE_DIR}/proto/BUILD.bazel"
  WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
)

execute_process(
  COMMAND ${CMAKE_COMMAND} -E copy_if_different
       "/home/yejie/SCS_lab/DyTO/DyTO/tensorstore/third_party/local_proto_mirror/src/proto/def.bzl"
       "${CMAKE_CURRENT_SOURCE_DIR}/proto/def.bzl"
  WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
)

execute_process(
  COMMAND ${CMAKE_COMMAND} -E copy_if_different
       "/home/yejie/SCS_lab/DyTO/DyTO/tensorstore/third_party/local_proto_mirror/src/proto/compiler.bzl"
       "${CMAKE_CURRENT_SOURCE_DIR}/proto/compiler.bzl"
  WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
)

execute_process(
  COMMAND ${CMAKE_COMMAND} -E copy_if_different
       "/home/yejie/SCS_lab/DyTO/DyTO/tensorstore/third_party/local_proto_mirror/src/lpm.BUILD.bazel"
       "${CMAKE_CURRENT_SOURCE_DIR}/go/BUILD.bazel"
  WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
)

execute_process(
  COMMAND ${CMAKE_COMMAND} -E copy_if_different
       "/home/yejie/SCS_lab/DyTO/DyTO/tensorstore/third_party/local_proto_mirror/src/go/def.bzl"
       "${CMAKE_CURRENT_SOURCE_DIR}/go/def.bzl"
  WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
)

execute_process(
  COMMAND ${CMAKE_COMMAND} -E copy_if_different
       "/home/yejie/SCS_lab/DyTO/DyTO/tensorstore/third_party/local_proto_mirror/src/opencensus/proto/resource/v1/lpm.BUILD.bazel"
       "${CMAKE_CURRENT_SOURCE_DIR}/opencensus/proto/resource/v1/BUILD.bazel"
  WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
)

execute_process(
  COMMAND ${CMAKE_COMMAND} -E copy_if_different
       "/home/yejie/SCS_lab/DyTO/DyTO/tensorstore/third_party/local_proto_mirror/src/opencensus/proto/resource/v1/resource.proto"
       "${CMAKE_CURRENT_SOURCE_DIR}/opencensus/proto/resource/v1/resource.proto"
  WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
)

execute_process(
  COMMAND ${CMAKE_COMMAND} -E copy_if_different
       "/home/yejie/SCS_lab/DyTO/DyTO/tensorstore/third_party/local_proto_mirror/src/opencensus/proto/trace/v1/lpm.BUILD.bazel"
       "${CMAKE_CURRENT_SOURCE_DIR}/opencensus/proto/trace/v1/BUILD.bazel"
  WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
)

execute_process(
  COMMAND ${CMAKE_COMMAND} -E copy_if_different
       "/home/yejie/SCS_lab/DyTO/DyTO/tensorstore/third_party/local_proto_mirror/src/opencensus/proto/trace/v1/trace.proto"
       "${CMAKE_CURRENT_SOURCE_DIR}/opencensus/proto/trace/v1/trace.proto"
  WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
)

execute_process(
  COMMAND ${CMAKE_COMMAND} -E copy_if_different
       "/home/yejie/SCS_lab/DyTO/DyTO/tensorstore/third_party/local_proto_mirror/src/opencensus/proto/trace/v1/trace_config.proto"
       "${CMAKE_CURRENT_SOURCE_DIR}/opencensus/proto/trace/v1/trace_config.proto"
  WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
)

execute_process(
  COMMAND ${CMAKE_COMMAND} -E copy_if_different
       "/home/yejie/SCS_lab/DyTO/DyTO/tensorstore/third_party/local_proto_mirror/src/lpm.BUILD.bazel"
       "${CMAKE_CURRENT_SOURCE_DIR}/bazel/BUILD.bazel"
  WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
)

execute_process(
  COMMAND ${CMAKE_COMMAND} -E copy_if_different
       "/home/yejie/SCS_lab/DyTO/DyTO/tensorstore/third_party/local_proto_mirror/src/bazel/pgv_proto_library.bzl"
       "${CMAKE_CURRENT_SOURCE_DIR}/bazel/pgv_proto_library.bzl"
  WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
)

execute_process(
  COMMAND ${CMAKE_COMMAND} -E copy_if_different
       "/home/yejie/SCS_lab/DyTO/DyTO/tensorstore/third_party/local_proto_mirror/src/validate/lpm.BUILD.bazel"
       "${CMAKE_CURRENT_SOURCE_DIR}/validate/BUILD.bazel"
  WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
)

execute_process(
  COMMAND ${CMAKE_COMMAND} -E copy_if_different
       "/home/yejie/SCS_lab/DyTO/DyTO/tensorstore/third_party/local_proto_mirror/src/validate/validate.proto"
       "${CMAKE_CURRENT_SOURCE_DIR}/validate/validate.proto"
  WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
)

project("lpm")
execute_process(
  COMMAND ${Python3_EXECUTABLE} "/home/yejie/SCS_lab/DyTO/DyTO/tensorstore/tools/cmake/run_bazel_to_cmake.py"
        --load-workspace "/home/yejie/SCS_lab/DyTO/DyTO/tensorstore/build/bazel_workspace_info.pickle"
        --cmake-project-name lpm
        --cmake-binary-dir "${CMAKE_CURRENT_BINARY_DIR}"
        --bazel-repo-name local_proto_mirror
        --build-rules-output "${CMAKE_CURRENT_BINARY_DIR}/build_rules.cmake"
  WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
  COMMAND_ERROR_IS_FATAL ANY)
include("${CMAKE_CURRENT_BINARY_DIR}/build_rules.cmake")
