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

#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "absl/status/status.h"
#include "absl/strings/cord.h"
#include <nlohmann/json.hpp>
#include "tensorstore/array.h"
#include "tensorstore/chunk_layout.h"
#include "tensorstore/codec_spec.h"
#include "tensorstore/data_type.h"
#include "tensorstore/driver/hdf5/compressor.h"
#include "tensorstore/index.h"
#include "tensorstore/index_space/dimension_units.h"
#include "tensorstore/index_space/index_domain.h"
#include "tensorstore/internal/json_binding/bindable.h"
#include "tensorstore/json_serialization_options_base.h"
#include "tensorstore/rank.h"
#include "tensorstore/schema.h"
#include "tensorstore/serialization/fwd.h"
#include "tensorstore/util/garbage_collection/fwd.h"
#include "tensorstore/util/result.h"
#include "tensorstore/util/span.h"

namespace tensorstore {
namespace internal_hdf5 {

  class HDF5Metadata {
  public:
      /// Length of `shape`, `axes` and `chunk_shape`.
      /// (Jie): not sure if we need the rank
      DimensionIndex rank = dynamic_rank;

      /// Specifies the current shape of the full volume.
      std::vector<Index> shape;

      /// Specifies the chunk size
      std::vector<Index> chunk_shape;

      Compressor compressor;
      DataType dtype;

      TENSORSTORE_DECLARE_JSON_DEFAULT_BINDER(HDF5Metadata,
                                              internal_json_binding::NoOptions,
                                              tensorstore::IncludeDefaults)
      std::string GetCompatibilityKey() const;
  };

/// Representation of partial metadata/metadata constraints specified as the
/// "metadata" member in the DriverSpec.
class HDF5MetadataConstraints {
 public:
  /// Length of `shape`, `axes` and `chunk_shape` if any are specified.  If none
  /// are specified, equal to `dynamic_rank`.
  DimensionIndex rank = dynamic_rank;

  /// Specifies the current shape of the full dataset.
  std::optional<std::vector<Index>> shape;

  /// Specifies the chunk size 
  std::optional<std::vector<Index>> chunk_shape;

  std::optional<Compressor> compressor;

  std::optional<DataType> dtype;

  TENSORSTORE_DECLARE_JSON_DEFAULT_BINDER(HDF5MetadataConstraints,
                                          internal_json_binding::NoOptions,
                                          tensorstore::IncludeDefaults)
};

class HDF5CodecSpec : public internal::CodecDriverSpec {
 public:
  constexpr static char id[] = "hdf5";

  CodecSpec Clone() const final;
  absl::Status DoMergeFrom(const internal::CodecDriverSpec& other_base) final;

  std::optional<Compressor> compressor;

  TENSORSTORE_DECLARE_JSON_DEFAULT_BINDER(HDF5CodecSpec, FromJsonOptions,
                                          ToJsonOptions,
                                          ::nlohmann::json::object_t)
};

/// Validates that `metadata` is consistent with `constraints`.
absl::Status ValidateMetadata(const HDF5Metadata& metadata,
                              const HDF5MetadataConstraints& constraints);

/// Converts `metadata_constraints` to a full metadata object.
///
/// \error `absl::StatusCode::kInvalidArgument` if any required fields are
///     unspecified.
Result<std::shared_ptr<const HDF5Metadata>> GetNewMetadata(
    const HDF5MetadataConstraints& metadata_constraints, const Schema& schema);

/// Validates that `schema` is compatible with `metadata`.
absl::Status ValidateMetadataSchema(const HDF5Metadata& metadata,
                                    const Schema& schema);


/// Returns the combined domain from `metadata_constraints` and `schema`.
///
/// If the domain is unspecified, returns a null domain.
///
/// \error `absl::StatusCode::kInvalidArgument` if `metadata_constraints` is
///     inconsistent with `schema`.
Result<IndexDomain<>> GetEffectiveDomain(
    const HDF5MetadataConstraints& metadata_constraints, const Schema& schema);

// /// Sets chunk layout constraints implied by `rank` and `chunk_shape`.
// absl::Status SetChunkLayoutFromMetadata(
//     DimensionIndex rank, std::optional<span<const Index>> chunk_shape,
//     ChunkLayout& chunk_layout);

/// Returns the combined chunk layout from `metadata_constraints` and `schema`.
///
/// \error `absl::StatusCode::kInvalidArgument` if `metadata_constraints` is
///     inconsistent with `schema`.
// Result<ChunkLayout> GetEffectiveChunkLayout(
//     const HDF5MetadataConstraints& metadata_constraints, const Schema& schema);

/// Returns the combined codec spec from `metadata_constraints` and `schema`.
///
/// \returns Non-null pointer.
/// \error `absl::StatusCode::kInvalidArgument` if `metadata_constraints` is
///     inconsistent with `schema`.
Result<internal::CodecDriverSpec::PtrT<HDF5CodecSpec>> GetEffectiveCodec(
    const HDF5MetadataConstraints& metadata_constraints, const Schema& schema);

/// Returns the codec from the specified metadata.
CodecSpec GetCodecFromMetadata(const HDF5Metadata& metadata);

/// Combines the units and resolution fields into a dimension units vector.
// DimensionUnitsVector GetDimensionUnits(
//     DimensionIndex metadata_rank,
//     const HDF5Metadata::UnitsAndResolution& units_and_resolution);

/// Returns the combined dimension units from `units_and_resolution` and
/// `schema_units`.
///
/// \error `absl::StatusCode::kInvalidArgument` if `units_and_resolution` is
///     inconsistent with `schema_units`.
// Result<DimensionUnitsVector> GetEffectiveDimensionUnits(
//     DimensionIndex metadata_rank,
//     const HDF5Metadata::UnitsAndResolution& units_and_resolution,
//     Schema::DimensionUnits schema_units);

/// Decodes a chunk.
///
/// The layout of the returned array is only valid as long as `metadata`.
// Result<SharedArray<const void>> DecodeChunk(const HDF5Metadata& metadata,
//                                             absl::Cord buffer);

// /// Encodes a chunk.
// Result<absl::Cord> EncodeChunk(const HDF5Metadata& metadata,
//                                SharedArrayView<const void> array);

/// Validates that `dtype` is supported by HDF5.
///
/// \dchecks `dtype.valid()`
absl::Status ValidateDataType(DataType dtype);

}  // namespace internal_hdf5
}  // namespace tensorstore

TENSORSTORE_DECLARE_SERIALIZER_SPECIALIZATION(
    tensorstore::internal_hdf5::HDF5MetadataConstraints)

TENSORSTORE_DECLARE_GARBAGE_COLLECTION_NOT_REQUIRED(
    tensorstore::internal_hdf5::HDF5MetadataConstraints)

#endif  // TENSORSTORE_DRIVER_HDF5_METADATA_H_