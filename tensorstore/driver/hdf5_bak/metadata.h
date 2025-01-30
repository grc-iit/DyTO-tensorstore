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

#ifndef TENSORSTORE_DRIVER_HDF5_METADATA_H_
#define TENSORSTORE_DRIVER_HDF5_METADATA_H_

/// \file
/// Decoded representation of HDF5 metadata.

#include <stddef.h>

#include <cstddef>  // std::nullptr_t
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "absl/container/inlined_vector.h"
#include "absl/status/status.h"
#include "absl/strings/cord.h"
#include <nlohmann/json.hpp>
#include "tensorstore/array.h"
#include "tensorstore/contiguous_layout.h"
#include "tensorstore/driver/hdf5/compressor.h"
#include "tensorstore/index.h"
#include "tensorstore/internal/json_binding/bindable.h"
#include "tensorstore/json_serialization_options_base.h"
#include "tensorstore/rank.h"
#include "tensorstore/serialization/fwd.h"
#include "tensorstore/strided_layout.h"
#include "tensorstore/util/garbage_collection/fwd.h"
#include "tensorstore/util/result.h"
#include "tensorstore/util/span.h"

namespace tensorstore {
namespace internal_hdf5 {

struct HDF5ChunkLayout {
    // (TODO): Add the fields in the ChunkLayout struct
};

/// Parse the HDF5 dataset metadata.
struct HDF5Metadata {

  DimensionIndex rank = dynamic_rank;

  /// The shape of array (current dataset).
  std::vector<Index> shape;

  // /// Chunk shape.  Must have same length as `shape`.
  // HDF5 chunks
  std::vector<Index> chunks;

  DataType dtype;
  // Compressor compressor;

  /// Encoded layout of chunk.
  ContiguousLayoutOrder order;
  /// Jie's Question: what is filters in HDF5 metadata?
  std::nullptr_t filters;

  /// Fill values for each of the fields.  Must have same length as
  /// `dtype.fields`.
  std::vector<SharedArray<const void>> fill_value;

  //::nlohmann::json::object_t extra_members;

  // Derived information computed from `dtype`, `order`, and `chunks`.

  // ZarrChunkLayout chunk_layout;
  // HDF5ChunkLayout chunk_layout;

  TENSORSTORE_DECLARE_JSON_DEFAULT_BINDER(HDF5Metadata,
                                          internal_json_binding::NoOptions,
                                          tensorstore::IncludeDefaults)

  /// Appends to `*out` a string that corresponds to the equivalence
  /// relationship defined by `IsMetadataCompatible`.
  friend void EncodeCacheKeyAdl(std::string* out, const HDF5Metadata& metadata);
};

/// Validates chunk layout and computes `metadata.chunk_layout`.
absl::Status ValidateMetadata(HDF5Metadata& metadata);

using HDF5MetadataPtr = std::shared_ptr<HDF5Metadata>;

/// Partially-specified hdf5 metadata used either to validate existing metadata
/// or to create a new array.
struct HDF5PartialMetadata {
  // The following members are common to both `HDF5Metadata` and
  // `HDF5PartialMetadata`, except that in `HDF5PartialMetadata` they are
  // `std::optional`-wrapped.

  DimensionIndex rank = dynamic_rank;

  /// Overall shape of array.
  std::optional<std::vector<Index>> shape;

  /// Chunk shape.  Must have same length as `shape`.
  std::optional<std::vector<Index>> chunks;

  std::optional<DataType> dtype;
  //std::optional<Compressor> compressor;

  /// Encoded layout of chunk.
  //std::optional<ContiguousLayoutOrder> order;
  std::optional<std::nullptr_t> filters;

  /// Fill values for each of the fields.  Must have same length as
  /// `dtype.fields`.
  std::optional<std::vector<SharedArray<const void>>> fill_value;

  TENSORSTORE_DECLARE_JSON_DEFAULT_BINDER(HDF5PartialMetadata,
                                          internal_json_binding::NoOptions,
                                          tensorstore::IncludeDefaults)
};

}  // namespace internal_hdf5
}  // namespace tensorstore

TENSORSTORE_DECLARE_SERIALIZER_SPECIALIZATION(
    tensorstore::internal_hdf5::HDF5PartialMetadata)

TENSORSTORE_DECLARE_GARBAGE_COLLECTION_NOT_REQUIRED(
    tensorstore::internal_hdf5::HDF5PartialMetadata)

#endif  // TENSORSTORE_DRIVER_HDF5_METADATA_H_
