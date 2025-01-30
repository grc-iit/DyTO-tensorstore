// Copyright 2020 The TensorStore Authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

/// \file
/// Defines the "zlib" and "gzip" compressors for zarr.  Linking in this
/// library automatically registers them.

/// (TODO): check if hdf5 has compressor, if not, delete the compressor

#include "tensorstore/driver/hdf5/compressor.h"
#include "tensorstore/driver/hdf5/compressor_registry.h"
#include "tensorstore/internal/compression/zlib_compressor.h"
#include "tensorstore/internal/json_binding/json_binding.h"

namespace tensorstore {
namespace internal_hdf5 {
namespace {

struct GzipCompressor : public internal::ZlibCompressor {};

struct Registration {
  Registration() {
    constexpr auto GetBinder = [](bool use_gzip_header) {
      namespace jb = tensorstore::internal_json_binding;
      return jb::Object(
          jb::Initialize(
              [=](auto* obj) { obj->use_gzip_header = use_gzip_header; }),
          jb::Member("level", jb::Projection(
                                  &internal::ZlibCompressor::level,
                                  jb::DefaultValue<jb::kAlwaysIncludeDefaults>(
                                      [](auto* v) { *v = 1; },
                                      jb::Integer<int>(0, 9)))));
    };
    RegisterCompressor<GzipCompressor>("gzip",
                                       GetBinder(/*use_gzip_header=*/true));
  }
} registration;

}  // namespace
}  // namespace internal_hdf5
}  // namespace tensorstore
