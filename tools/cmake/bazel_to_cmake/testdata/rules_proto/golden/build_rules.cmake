find_package(Protobuf REQUIRED)

# proto_library(@rules_proto_test_repo//:a_proto)
add_library(CMakeProject_a_proto INTERFACE)
target_sources(CMakeProject_a_proto INTERFACE
        "${PROJECT_SOURCE_DIR}/a.proto")
target_include_directories(CMakeProject_a_proto INTERFACE
       "${PROJECT_SOURCE_DIR}")
add_library(CMakeProject::a_proto ALIAS CMakeProject_a_proto)

# @rules_proto_test_repo//:a_proto__cpp_library
add_library(CMakeProject_a_proto__cpp_library)
set_property(TARGET CMakeProject_a_proto__cpp_library PROPERTY LINKER_LANGUAGE "CXX")
target_link_libraries(CMakeProject_a_proto__cpp_library PUBLIC
        "protobuf::libprotobuf")
target_compile_features(CMakeProject_a_proto__cpp_library PUBLIC cxx_std_17)
add_library(CMakeProject::a_proto__cpp_library ALIAS CMakeProject_a_proto__cpp_library)

btc_protobuf(
    TARGET CMakeProject_a_proto__cpp_library
    PROTO_TARGET CMakeProject_a_proto
    LANGUAGE cpp
    GENERATE_EXTENSIONS ".pb.h" ".pb.cc"
    PROTOC_OPTIONS --experimental_allow_proto3_optional
    PROTOC_OUT_DIR ${PROJECT_BINARY_DIR}/7647136c
    DEPENDS "protobuf::protoc"
)

# @rules_proto_test_repo//:a_proto__minitable_library
add_library(CMakeProject_a_proto__minitable_library)
set_property(TARGET CMakeProject_a_proto__minitable_library PROPERTY LINKER_LANGUAGE "CXX")
target_link_libraries(CMakeProject_a_proto__minitable_library PUBLIC
        "protobuf::upb_generated_code_support__only_for_generated_code_do_not_use__i_give_permission_to_break_me")
target_compile_features(CMakeProject_a_proto__minitable_library PUBLIC cxx_std_17)
add_library(CMakeProject::a_proto__minitable_library ALIAS CMakeProject_a_proto__minitable_library)

btc_protobuf(
    TARGET CMakeProject_a_proto__minitable_library
    PROTO_TARGET CMakeProject_a_proto
    LANGUAGE upb_minitable
    GENERATE_EXTENSIONS ".upb_minitable.h" ".upb_minitable.c"
    PROTOC_OPTIONS --experimental_allow_proto3_optional
    PROTOC_OUT_DIR ${PROJECT_BINARY_DIR}/7647136c
    PLUGIN protoc-gen-upb_minitable=$<TARGET_FILE:protobuf::protoc_gen_upb_minitable_stage1>
    DEPENDS "protobuf::protoc" "protobuf::protoc_gen_upb_minitable_stage1"
)

# @rules_proto_test_repo//:a_proto__upb_library
add_library(CMakeProject_a_proto__upb_library)
set_property(TARGET CMakeProject_a_proto__upb_library PROPERTY LINKER_LANGUAGE "CXX")
target_link_libraries(CMakeProject_a_proto__upb_library PUBLIC
        "CMakeProject::a_proto__minitable_library"
        "protobuf::upb_generated_code_support__only_for_generated_code_do_not_use__i_give_permission_to_break_me")
target_compile_features(CMakeProject_a_proto__upb_library PUBLIC cxx_std_17)
add_library(CMakeProject::a_proto__upb_library ALIAS CMakeProject_a_proto__upb_library)

btc_protobuf(
    TARGET CMakeProject_a_proto__upb_library
    PROTO_TARGET CMakeProject_a_proto
    LANGUAGE upb
    GENERATE_EXTENSIONS ".upb.h" ".upb.c"
    PROTOC_OPTIONS --experimental_allow_proto3_optional
    PROTOC_OUT_DIR ${PROJECT_BINARY_DIR}/7647136c
    PLUGIN protoc-gen-upb=$<TARGET_FILE:protobuf::protoc_gen_upb>
    DEPENDS "protobuf::protoc" "protobuf::protoc_gen_upb"
)

# @rules_proto_test_repo//:a_proto__upbdefs_library
add_library(CMakeProject_a_proto__upbdefs_library)
set_property(TARGET CMakeProject_a_proto__upbdefs_library PROPERTY LINKER_LANGUAGE "CXX")
target_link_libraries(CMakeProject_a_proto__upbdefs_library PUBLIC
        "CMakeProject::a_proto__minitable_library"
        "Protobuf::upb_generated_reflection_support__only_for_generated_code_do_not_use__i_give_permission_to_break_me"
        "Protobuf::upb_port")
target_compile_features(CMakeProject_a_proto__upbdefs_library PUBLIC cxx_std_17)
add_library(CMakeProject::a_proto__upbdefs_library ALIAS CMakeProject_a_proto__upbdefs_library)

btc_protobuf(
    TARGET CMakeProject_a_proto__upbdefs_library
    PROTO_TARGET CMakeProject_a_proto
    LANGUAGE upbdefs
    GENERATE_EXTENSIONS ".upbdefs.h" ".upbdefs.c"
    PROTOC_OPTIONS --experimental_allow_proto3_optional
    PROTOC_OUT_DIR ${PROJECT_BINARY_DIR}/7647136c
    PLUGIN protoc-gen-upbdefs=$<TARGET_FILE:protobuf::protoc_gen_upbdefs>
    DEPENDS "protobuf::protoc" "protobuf::protoc_gen_upbdefs"
)

# proto_library(@rules_proto_test_repo//:ab_proto)
add_library(CMakeProject_ab_proto INTERFACE)
target_sources(CMakeProject_ab_proto INTERFACE
        "${PROJECT_SOURCE_DIR}/a.proto"
        "${PROJECT_SOURCE_DIR}/b.proto")
target_include_directories(CMakeProject_ab_proto INTERFACE
       "${PROJECT_SOURCE_DIR}")
add_library(CMakeProject::ab_proto ALIAS CMakeProject_ab_proto)

# @rules_proto_test_repo//:ab_proto__cpp_library
add_library(CMakeProject_ab_proto__cpp_library)
set_property(TARGET CMakeProject_ab_proto__cpp_library PROPERTY LINKER_LANGUAGE "CXX")
target_link_libraries(CMakeProject_ab_proto__cpp_library PUBLIC
        "protobuf::libprotobuf")
target_compile_features(CMakeProject_ab_proto__cpp_library PUBLIC cxx_std_17)
add_library(CMakeProject::ab_proto__cpp_library ALIAS CMakeProject_ab_proto__cpp_library)

btc_protobuf(
    TARGET CMakeProject_ab_proto__cpp_library
    PROTO_TARGET CMakeProject_ab_proto
    LANGUAGE cpp
    GENERATE_EXTENSIONS ".pb.h" ".pb.cc"
    PROTOC_OPTIONS --experimental_allow_proto3_optional
    PROTOC_OUT_DIR ${PROJECT_BINARY_DIR}/3ec71571
    DEPENDS "protobuf::protoc"
)

# @rules_proto_test_repo//:ab_proto__minitable_library
add_library(CMakeProject_ab_proto__minitable_library)
set_property(TARGET CMakeProject_ab_proto__minitable_library PROPERTY LINKER_LANGUAGE "CXX")
target_link_libraries(CMakeProject_ab_proto__minitable_library PUBLIC
        "protobuf::upb_generated_code_support__only_for_generated_code_do_not_use__i_give_permission_to_break_me")
target_compile_features(CMakeProject_ab_proto__minitable_library PUBLIC cxx_std_17)
add_library(CMakeProject::ab_proto__minitable_library ALIAS CMakeProject_ab_proto__minitable_library)

btc_protobuf(
    TARGET CMakeProject_ab_proto__minitable_library
    PROTO_TARGET CMakeProject_ab_proto
    LANGUAGE upb_minitable
    GENERATE_EXTENSIONS ".upb_minitable.h" ".upb_minitable.c"
    PROTOC_OPTIONS --experimental_allow_proto3_optional
    PROTOC_OUT_DIR ${PROJECT_BINARY_DIR}/3ec71571
    PLUGIN protoc-gen-upb_minitable=$<TARGET_FILE:protobuf::protoc_gen_upb_minitable_stage1>
    DEPENDS "protobuf::protoc" "protobuf::protoc_gen_upb_minitable_stage1"
)

# @rules_proto_test_repo//:ab_proto__upb_library
add_library(CMakeProject_ab_proto__upb_library)
set_property(TARGET CMakeProject_ab_proto__upb_library PROPERTY LINKER_LANGUAGE "CXX")
target_link_libraries(CMakeProject_ab_proto__upb_library PUBLIC
        "CMakeProject::ab_proto__minitable_library"
        "protobuf::upb_generated_code_support__only_for_generated_code_do_not_use__i_give_permission_to_break_me")
target_compile_features(CMakeProject_ab_proto__upb_library PUBLIC cxx_std_17)
add_library(CMakeProject::ab_proto__upb_library ALIAS CMakeProject_ab_proto__upb_library)

btc_protobuf(
    TARGET CMakeProject_ab_proto__upb_library
    PROTO_TARGET CMakeProject_ab_proto
    LANGUAGE upb
    GENERATE_EXTENSIONS ".upb.h" ".upb.c"
    PROTOC_OPTIONS --experimental_allow_proto3_optional
    PROTOC_OUT_DIR ${PROJECT_BINARY_DIR}/3ec71571
    PLUGIN protoc-gen-upb=$<TARGET_FILE:protobuf::protoc_gen_upb>
    DEPENDS "protobuf::protoc" "protobuf::protoc_gen_upb"
)

# @rules_proto_test_repo//:ab_proto__upbdefs_library
add_library(CMakeProject_ab_proto__upbdefs_library)
set_property(TARGET CMakeProject_ab_proto__upbdefs_library PROPERTY LINKER_LANGUAGE "CXX")
target_link_libraries(CMakeProject_ab_proto__upbdefs_library PUBLIC
        "CMakeProject::ab_proto__minitable_library"
        "Protobuf::upb_generated_reflection_support__only_for_generated_code_do_not_use__i_give_permission_to_break_me"
        "Protobuf::upb_port")
target_compile_features(CMakeProject_ab_proto__upbdefs_library PUBLIC cxx_std_17)
add_library(CMakeProject::ab_proto__upbdefs_library ALIAS CMakeProject_ab_proto__upbdefs_library)

btc_protobuf(
    TARGET CMakeProject_ab_proto__upbdefs_library
    PROTO_TARGET CMakeProject_ab_proto
    LANGUAGE upbdefs
    GENERATE_EXTENSIONS ".upbdefs.h" ".upbdefs.c"
    PROTOC_OPTIONS --experimental_allow_proto3_optional
    PROTOC_OUT_DIR ${PROJECT_BINARY_DIR}/3ec71571
    PLUGIN protoc-gen-upbdefs=$<TARGET_FILE:protobuf::protoc_gen_upbdefs>
    DEPENDS "protobuf::protoc" "protobuf::protoc_gen_upbdefs"
)

# cc_proto_library(@rules_proto_test_repo//:ab_protos_cc)
add_library(CMakeProject_ab_protos_cc INTERFACE)
target_link_libraries(CMakeProject_ab_protos_cc INTERFACE
        "CMakeProject::ab_proto__cpp_library")
target_compile_features(CMakeProject_ab_protos_cc INTERFACE cxx_std_17)
add_library(CMakeProject::ab_protos_cc ALIAS CMakeProject_ab_protos_cc)

# proto_library(@rules_proto_test_repo//:abc_proto)
add_library(CMakeProject_abc_proto INTERFACE)
target_sources(CMakeProject_abc_proto INTERFACE
        "${PROJECT_SOURCE_DIR}/c.proto")
target_link_libraries(CMakeProject_abc_proto INTERFACE
        "CMakeProject::ab_proto")
target_include_directories(CMakeProject_abc_proto INTERFACE
       "${PROJECT_SOURCE_DIR}")
add_library(CMakeProject::abc_proto ALIAS CMakeProject_abc_proto)

# @rules_proto_test_repo//:abc_proto__cpp_library
add_library(CMakeProject_abc_proto__cpp_library)
set_property(TARGET CMakeProject_abc_proto__cpp_library PROPERTY LINKER_LANGUAGE "CXX")
target_link_libraries(CMakeProject_abc_proto__cpp_library PUBLIC
        "CMakeProject::ab_proto__cpp_library"
        "protobuf::libprotobuf")
target_compile_features(CMakeProject_abc_proto__cpp_library PUBLIC cxx_std_17)
add_library(CMakeProject::abc_proto__cpp_library ALIAS CMakeProject_abc_proto__cpp_library)

btc_protobuf(
    TARGET CMakeProject_abc_proto__cpp_library
    PROTO_TARGET CMakeProject_abc_proto
    LANGUAGE cpp
    GENERATE_EXTENSIONS ".pb.h" ".pb.cc"
    PROTOC_OPTIONS --experimental_allow_proto3_optional
    PROTOC_OUT_DIR ${PROJECT_BINARY_DIR}/cbb28122
    DEPENDS "protobuf::protoc"
)

# @rules_proto_test_repo//:abc_proto__minitable_library
add_library(CMakeProject_abc_proto__minitable_library)
set_property(TARGET CMakeProject_abc_proto__minitable_library PROPERTY LINKER_LANGUAGE "CXX")
target_link_libraries(CMakeProject_abc_proto__minitable_library PUBLIC
        "CMakeProject::ab_proto__minitable_library"
        "protobuf::upb_generated_code_support__only_for_generated_code_do_not_use__i_give_permission_to_break_me")
target_compile_features(CMakeProject_abc_proto__minitable_library PUBLIC cxx_std_17)
add_library(CMakeProject::abc_proto__minitable_library ALIAS CMakeProject_abc_proto__minitable_library)

btc_protobuf(
    TARGET CMakeProject_abc_proto__minitable_library
    PROTO_TARGET CMakeProject_abc_proto
    LANGUAGE upb_minitable
    GENERATE_EXTENSIONS ".upb_minitable.h" ".upb_minitable.c"
    PROTOC_OPTIONS --experimental_allow_proto3_optional
    PROTOC_OUT_DIR ${PROJECT_BINARY_DIR}/cbb28122
    PLUGIN protoc-gen-upb_minitable=$<TARGET_FILE:protobuf::protoc_gen_upb_minitable_stage1>
    DEPENDS "protobuf::protoc" "protobuf::protoc_gen_upb_minitable_stage1"
)

# @rules_proto_test_repo//:abc_proto__upb_library
add_library(CMakeProject_abc_proto__upb_library)
set_property(TARGET CMakeProject_abc_proto__upb_library PROPERTY LINKER_LANGUAGE "CXX")
target_link_libraries(CMakeProject_abc_proto__upb_library PUBLIC
        "CMakeProject::ab_proto__minitable_library"
        "CMakeProject::ab_proto__upb_library"
        "CMakeProject::abc_proto__minitable_library"
        "protobuf::upb_generated_code_support__only_for_generated_code_do_not_use__i_give_permission_to_break_me")
target_compile_features(CMakeProject_abc_proto__upb_library PUBLIC cxx_std_17)
add_library(CMakeProject::abc_proto__upb_library ALIAS CMakeProject_abc_proto__upb_library)

btc_protobuf(
    TARGET CMakeProject_abc_proto__upb_library
    PROTO_TARGET CMakeProject_abc_proto
    LANGUAGE upb
    GENERATE_EXTENSIONS ".upb.h" ".upb.c"
    PROTOC_OPTIONS --experimental_allow_proto3_optional
    PROTOC_OUT_DIR ${PROJECT_BINARY_DIR}/cbb28122
    PLUGIN protoc-gen-upb=$<TARGET_FILE:protobuf::protoc_gen_upb>
    DEPENDS "protobuf::protoc" "protobuf::protoc_gen_upb"
)

# @rules_proto_test_repo//:abc_proto__upbdefs_library
add_library(CMakeProject_abc_proto__upbdefs_library)
set_property(TARGET CMakeProject_abc_proto__upbdefs_library PROPERTY LINKER_LANGUAGE "CXX")
target_link_libraries(CMakeProject_abc_proto__upbdefs_library PUBLIC
        "CMakeProject::ab_proto__minitable_library"
        "CMakeProject::ab_proto__upbdefs_library"
        "CMakeProject::abc_proto__minitable_library"
        "Protobuf::upb_generated_reflection_support__only_for_generated_code_do_not_use__i_give_permission_to_break_me"
        "Protobuf::upb_port")
target_compile_features(CMakeProject_abc_proto__upbdefs_library PUBLIC cxx_std_17)
add_library(CMakeProject::abc_proto__upbdefs_library ALIAS CMakeProject_abc_proto__upbdefs_library)

btc_protobuf(
    TARGET CMakeProject_abc_proto__upbdefs_library
    PROTO_TARGET CMakeProject_abc_proto
    LANGUAGE upbdefs
    GENERATE_EXTENSIONS ".upbdefs.h" ".upbdefs.c"
    PROTOC_OPTIONS --experimental_allow_proto3_optional
    PROTOC_OUT_DIR ${PROJECT_BINARY_DIR}/cbb28122
    PLUGIN protoc-gen-upbdefs=$<TARGET_FILE:protobuf::protoc_gen_upbdefs>
    DEPENDS "protobuf::protoc" "protobuf::protoc_gen_upbdefs"
)

# cc_proto_library(@rules_proto_test_repo//:abc_protos_cc)
add_library(CMakeProject_abc_protos_cc INTERFACE)
target_link_libraries(CMakeProject_abc_protos_cc INTERFACE
        "CMakeProject::abc_proto__cpp_library")
target_compile_features(CMakeProject_abc_protos_cc INTERFACE cxx_std_17)
add_library(CMakeProject::abc_protos_cc ALIAS CMakeProject_abc_protos_cc)

# proto_library(@rules_proto_test_repo//:b_proto)
add_library(CMakeProject_b_proto INTERFACE)
target_sources(CMakeProject_b_proto INTERFACE
        "${PROJECT_SOURCE_DIR}/b.proto")
target_include_directories(CMakeProject_b_proto INTERFACE
       "${PROJECT_SOURCE_DIR}")
add_library(CMakeProject::b_proto ALIAS CMakeProject_b_proto)

# @rules_proto_test_repo//:b_proto__cpp_library
add_library(CMakeProject_b_proto__cpp_library)
set_property(TARGET CMakeProject_b_proto__cpp_library PROPERTY LINKER_LANGUAGE "CXX")
target_link_libraries(CMakeProject_b_proto__cpp_library PUBLIC
        "protobuf::libprotobuf")
target_compile_features(CMakeProject_b_proto__cpp_library PUBLIC cxx_std_17)
add_library(CMakeProject::b_proto__cpp_library ALIAS CMakeProject_b_proto__cpp_library)

btc_protobuf(
    TARGET CMakeProject_b_proto__cpp_library
    PROTO_TARGET CMakeProject_b_proto
    LANGUAGE cpp
    GENERATE_EXTENSIONS ".pb.h" ".pb.cc"
    PROTOC_OPTIONS --experimental_allow_proto3_optional
    PROTOC_OUT_DIR ${PROJECT_BINARY_DIR}/0cdcec13
    DEPENDS "protobuf::protoc"
)

# @rules_proto_test_repo//:b_proto__minitable_library
add_library(CMakeProject_b_proto__minitable_library)
set_property(TARGET CMakeProject_b_proto__minitable_library PROPERTY LINKER_LANGUAGE "CXX")
target_link_libraries(CMakeProject_b_proto__minitable_library PUBLIC
        "protobuf::upb_generated_code_support__only_for_generated_code_do_not_use__i_give_permission_to_break_me")
target_compile_features(CMakeProject_b_proto__minitable_library PUBLIC cxx_std_17)
add_library(CMakeProject::b_proto__minitable_library ALIAS CMakeProject_b_proto__minitable_library)

btc_protobuf(
    TARGET CMakeProject_b_proto__minitable_library
    PROTO_TARGET CMakeProject_b_proto
    LANGUAGE upb_minitable
    GENERATE_EXTENSIONS ".upb_minitable.h" ".upb_minitable.c"
    PROTOC_OPTIONS --experimental_allow_proto3_optional
    PROTOC_OUT_DIR ${PROJECT_BINARY_DIR}/0cdcec13
    PLUGIN protoc-gen-upb_minitable=$<TARGET_FILE:protobuf::protoc_gen_upb_minitable_stage1>
    DEPENDS "protobuf::protoc" "protobuf::protoc_gen_upb_minitable_stage1"
)

# @rules_proto_test_repo//:b_proto__upb_library
add_library(CMakeProject_b_proto__upb_library)
set_property(TARGET CMakeProject_b_proto__upb_library PROPERTY LINKER_LANGUAGE "CXX")
target_link_libraries(CMakeProject_b_proto__upb_library PUBLIC
        "CMakeProject::b_proto__minitable_library"
        "protobuf::upb_generated_code_support__only_for_generated_code_do_not_use__i_give_permission_to_break_me")
target_compile_features(CMakeProject_b_proto__upb_library PUBLIC cxx_std_17)
add_library(CMakeProject::b_proto__upb_library ALIAS CMakeProject_b_proto__upb_library)

btc_protobuf(
    TARGET CMakeProject_b_proto__upb_library
    PROTO_TARGET CMakeProject_b_proto
    LANGUAGE upb
    GENERATE_EXTENSIONS ".upb.h" ".upb.c"
    PROTOC_OPTIONS --experimental_allow_proto3_optional
    PROTOC_OUT_DIR ${PROJECT_BINARY_DIR}/0cdcec13
    PLUGIN protoc-gen-upb=$<TARGET_FILE:protobuf::protoc_gen_upb>
    DEPENDS "protobuf::protoc" "protobuf::protoc_gen_upb"
)

# @rules_proto_test_repo//:b_proto__upbdefs_library
add_library(CMakeProject_b_proto__upbdefs_library)
set_property(TARGET CMakeProject_b_proto__upbdefs_library PROPERTY LINKER_LANGUAGE "CXX")
target_link_libraries(CMakeProject_b_proto__upbdefs_library PUBLIC
        "CMakeProject::b_proto__minitable_library"
        "Protobuf::upb_generated_reflection_support__only_for_generated_code_do_not_use__i_give_permission_to_break_me"
        "Protobuf::upb_port")
target_compile_features(CMakeProject_b_proto__upbdefs_library PUBLIC cxx_std_17)
add_library(CMakeProject::b_proto__upbdefs_library ALIAS CMakeProject_b_proto__upbdefs_library)

btc_protobuf(
    TARGET CMakeProject_b_proto__upbdefs_library
    PROTO_TARGET CMakeProject_b_proto
    LANGUAGE upbdefs
    GENERATE_EXTENSIONS ".upbdefs.h" ".upbdefs.c"
    PROTOC_OPTIONS --experimental_allow_proto3_optional
    PROTOC_OUT_DIR ${PROJECT_BINARY_DIR}/0cdcec13
    PLUGIN protoc-gen-upbdefs=$<TARGET_FILE:protobuf::protoc_gen_upbdefs>
    DEPENDS "protobuf::protoc" "protobuf::protoc_gen_upbdefs"
)

# proto_library(@rules_proto_test_repo//:x_proto)
add_library(CMakeProject_x_proto INTERFACE)
target_sources(CMakeProject_x_proto INTERFACE
        "${PROJECT_SOURCE_DIR}/x.proto")
target_link_libraries(CMakeProject_x_proto INTERFACE
        "Protobuf_any_proto")
target_include_directories(CMakeProject_x_proto INTERFACE
       ${Protobuf_IMPORT_DIRS}
       "${PROJECT_SOURCE_DIR}")
add_library(CMakeProject::x_proto ALIAS CMakeProject_x_proto)

# @rules_proto_test_repo//:x_proto__minitable_library
add_library(CMakeProject_x_proto__minitable_library)
set_property(TARGET CMakeProject_x_proto__minitable_library PROPERTY LINKER_LANGUAGE "CXX")
target_link_libraries(CMakeProject_x_proto__minitable_library PUBLIC
        "Protobuf::any_proto__minitable_library"
        "protobuf::upb_generated_code_support__only_for_generated_code_do_not_use__i_give_permission_to_break_me")
target_compile_features(CMakeProject_x_proto__minitable_library PUBLIC cxx_std_17)
add_library(CMakeProject::x_proto__minitable_library ALIAS CMakeProject_x_proto__minitable_library)

btc_protobuf(
    TARGET CMakeProject_x_proto__minitable_library
    PROTO_TARGET CMakeProject_x_proto
    LANGUAGE upb_minitable
    GENERATE_EXTENSIONS ".upb_minitable.h" ".upb_minitable.c"
    PROTOC_OPTIONS --experimental_allow_proto3_optional
    PROTOC_OUT_DIR ${PROJECT_BINARY_DIR}/db1b3834
    PLUGIN protoc-gen-upb_minitable=$<TARGET_FILE:protobuf::protoc_gen_upb_minitable_stage1>
    DEPENDS "protobuf::protoc" "protobuf::protoc_gen_upb_minitable_stage1"
)

# @rules_proto_test_repo//:x_proto__upb_library
add_library(CMakeProject_x_proto__upb_library)
set_property(TARGET CMakeProject_x_proto__upb_library PROPERTY LINKER_LANGUAGE "CXX")
target_link_libraries(CMakeProject_x_proto__upb_library PUBLIC
        "CMakeProject::x_proto__minitable_library"
        "Protobuf::any_proto__minitable_library"
        "Protobuf::any_proto__upb_library"
        "protobuf::upb_generated_code_support__only_for_generated_code_do_not_use__i_give_permission_to_break_me")
target_compile_features(CMakeProject_x_proto__upb_library PUBLIC cxx_std_17)
add_library(CMakeProject::x_proto__upb_library ALIAS CMakeProject_x_proto__upb_library)

btc_protobuf(
    TARGET CMakeProject_x_proto__upb_library
    PROTO_TARGET CMakeProject_x_proto
    LANGUAGE upb
    GENERATE_EXTENSIONS ".upb.h" ".upb.c"
    PROTOC_OPTIONS --experimental_allow_proto3_optional
    PROTOC_OUT_DIR ${PROJECT_BINARY_DIR}/db1b3834
    PLUGIN protoc-gen-upb=$<TARGET_FILE:protobuf::protoc_gen_upb>
    DEPENDS "protobuf::protoc" "protobuf::protoc_gen_upb"
)

# @rules_proto_test_repo//:x_proto__upbdefs_library
add_library(CMakeProject_x_proto__upbdefs_library)
set_property(TARGET CMakeProject_x_proto__upbdefs_library PROPERTY LINKER_LANGUAGE "CXX")
target_link_libraries(CMakeProject_x_proto__upbdefs_library PUBLIC
        "CMakeProject::x_proto__minitable_library"
        "Protobuf::any_proto__minitable_library"
        "Protobuf::any_proto__upbdefs_library"
        "Protobuf::upb_generated_reflection_support__only_for_generated_code_do_not_use__i_give_permission_to_break_me"
        "Protobuf::upb_port")
target_compile_features(CMakeProject_x_proto__upbdefs_library PUBLIC cxx_std_17)
add_library(CMakeProject::x_proto__upbdefs_library ALIAS CMakeProject_x_proto__upbdefs_library)

btc_protobuf(
    TARGET CMakeProject_x_proto__upbdefs_library
    PROTO_TARGET CMakeProject_x_proto
    LANGUAGE upbdefs
    GENERATE_EXTENSIONS ".upbdefs.h" ".upbdefs.c"
    PROTOC_OPTIONS --experimental_allow_proto3_optional
    PROTOC_OUT_DIR ${PROJECT_BINARY_DIR}/db1b3834
    PLUGIN protoc-gen-upbdefs=$<TARGET_FILE:protobuf::protoc_gen_upbdefs>
    DEPENDS "protobuf::protoc" "protobuf::protoc_gen_upbdefs"
)

# @rules_proto_test_repo//:x_proto__cpp_library
add_library(CMakeProject_x_proto__cpp_library)
set_property(TARGET CMakeProject_x_proto__cpp_library PROPERTY LINKER_LANGUAGE "CXX")
target_link_libraries(CMakeProject_x_proto__cpp_library PUBLIC
        "Protobuf::any_proto__cpp_library"
        "protobuf::libprotobuf")
target_compile_features(CMakeProject_x_proto__cpp_library PUBLIC cxx_std_17)
add_library(CMakeProject::x_proto__cpp_library ALIAS CMakeProject_x_proto__cpp_library)

btc_protobuf(
    TARGET CMakeProject_x_proto__cpp_library
    PROTO_TARGET CMakeProject_x_proto
    LANGUAGE cpp
    GENERATE_EXTENSIONS ".pb.h" ".pb.cc"
    PROTOC_OPTIONS --experimental_allow_proto3_optional
    PROTOC_OUT_DIR ${PROJECT_BINARY_DIR}/db1b3834
    DEPENDS "protobuf::protoc"
)

# cc_proto_library(@rules_proto_test_repo//:x_proto_cc)
add_library(CMakeProject_x_proto_cc INTERFACE)
target_link_libraries(CMakeProject_x_proto_cc INTERFACE
        "CMakeProject::x_proto__cpp_library")
target_compile_features(CMakeProject_x_proto_cc INTERFACE cxx_std_17)
add_library(CMakeProject::x_proto_cc ALIAS CMakeProject_x_proto_cc)

# proto_library(@rules_proto_test_repo//:y_proto)
add_library(CMakeProject_y_proto INTERFACE)
target_sources(CMakeProject_y_proto INTERFACE
        "${PROJECT_SOURCE_DIR}/src/subdir/y.proto")
target_include_directories(CMakeProject_y_proto INTERFACE
       "${PROJECT_SOURCE_DIR}/src")
add_library(CMakeProject::y_proto ALIAS CMakeProject_y_proto)

# @rules_proto_test_repo//:y_proto__cpp_library
add_library(CMakeProject_y_proto__cpp_library)
set_property(TARGET CMakeProject_y_proto__cpp_library PROPERTY LINKER_LANGUAGE "CXX")
target_link_libraries(CMakeProject_y_proto__cpp_library PUBLIC
        "protobuf::libprotobuf")
target_compile_features(CMakeProject_y_proto__cpp_library PUBLIC cxx_std_17)
add_library(CMakeProject::y_proto__cpp_library ALIAS CMakeProject_y_proto__cpp_library)

btc_protobuf(
    TARGET CMakeProject_y_proto__cpp_library
    PROTO_TARGET CMakeProject_y_proto
    LANGUAGE cpp
    GENERATE_EXTENSIONS ".pb.h" ".pb.cc"
    PROTOC_OPTIONS --experimental_allow_proto3_optional
    PROTOC_OUT_DIR ${PROJECT_BINARY_DIR}/573e9085/src
    DEPENDS "protobuf::protoc"
)

# @rules_proto_test_repo//:y_proto__minitable_library
add_library(CMakeProject_y_proto__minitable_library)
set_property(TARGET CMakeProject_y_proto__minitable_library PROPERTY LINKER_LANGUAGE "CXX")
target_link_libraries(CMakeProject_y_proto__minitable_library PUBLIC
        "protobuf::upb_generated_code_support__only_for_generated_code_do_not_use__i_give_permission_to_break_me")
target_compile_features(CMakeProject_y_proto__minitable_library PUBLIC cxx_std_17)
add_library(CMakeProject::y_proto__minitable_library ALIAS CMakeProject_y_proto__minitable_library)

btc_protobuf(
    TARGET CMakeProject_y_proto__minitable_library
    PROTO_TARGET CMakeProject_y_proto
    LANGUAGE upb_minitable
    GENERATE_EXTENSIONS ".upb_minitable.h" ".upb_minitable.c"
    PROTOC_OPTIONS --experimental_allow_proto3_optional
    PROTOC_OUT_DIR ${PROJECT_BINARY_DIR}/573e9085/src
    PLUGIN protoc-gen-upb_minitable=$<TARGET_FILE:protobuf::protoc_gen_upb_minitable_stage1>
    DEPENDS "protobuf::protoc" "protobuf::protoc_gen_upb_minitable_stage1"
)

# @rules_proto_test_repo//:y_proto__upb_library
add_library(CMakeProject_y_proto__upb_library)
set_property(TARGET CMakeProject_y_proto__upb_library PROPERTY LINKER_LANGUAGE "CXX")
target_link_libraries(CMakeProject_y_proto__upb_library PUBLIC
        "CMakeProject::y_proto__minitable_library"
        "protobuf::upb_generated_code_support__only_for_generated_code_do_not_use__i_give_permission_to_break_me")
target_compile_features(CMakeProject_y_proto__upb_library PUBLIC cxx_std_17)
add_library(CMakeProject::y_proto__upb_library ALIAS CMakeProject_y_proto__upb_library)

btc_protobuf(
    TARGET CMakeProject_y_proto__upb_library
    PROTO_TARGET CMakeProject_y_proto
    LANGUAGE upb
    GENERATE_EXTENSIONS ".upb.h" ".upb.c"
    PROTOC_OPTIONS --experimental_allow_proto3_optional
    PROTOC_OUT_DIR ${PROJECT_BINARY_DIR}/573e9085/src
    PLUGIN protoc-gen-upb=$<TARGET_FILE:protobuf::protoc_gen_upb>
    DEPENDS "protobuf::protoc" "protobuf::protoc_gen_upb"
)
