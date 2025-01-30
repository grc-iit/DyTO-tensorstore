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

#include "tensorstore/driver/hdf5/metadata.h"

#include <stddef.h>
#include <stdint.h>

#include <algorithm>
#include <array>
#include <cassert>
#include <memory>
#include <optional>
#include <string>
#include <utility>

#include "absl/algorithm/container.h"
#include "absl/meta/type_traits.h"
#include "absl/status/status.h"
#include "absl/strings/cord.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include <nlohmann/json.hpp>
#include "riegeli/bytes/cord_reader.h"
#include "riegeli/bytes/cord_writer.h"
#include "riegeli/bytes/reader.h"
#include "riegeli/bytes/writer.h"
#include "riegeli/endian/endian_reading.h"
#include "riegeli/endian/endian_writing.h"
#include "tensorstore/array.h"
#include "tensorstore/box.h"
#include "tensorstore/chunk_layout.h"
#include "tensorstore/codec_spec.h"
#include "tensorstore/codec_spec_registry.h"
#include "tensorstore/contiguous_layout.h"
#include "tensorstore/data_type.h"
#include "tensorstore/driver/hdf5/compressor.h"
#include "tensorstore/index.h"
#include "tensorstore/index_space/dimension_units.h"
#include "tensorstore/index_space/index_domain.h"
#include "tensorstore/index_space/index_domain_builder.h"
#include "tensorstore/internal/json/same.h"
#include "tensorstore/internal/json_binding/bindable.h"
#include "tensorstore/internal/json_binding/data_type.h"
#include "tensorstore/internal/json_binding/dimension_indexed.h"
#include "tensorstore/internal/json_binding/json_binding.h"
#include "tensorstore/internal/json_binding/std_optional.h"
#include "tensorstore/internal/json_metadata_matching.h"
#include "tensorstore/internal/riegeli/array_endian_codec.h"
#include "tensorstore/internal/type_traits.h"
#include "tensorstore/rank.h"
#include "tensorstore/schema.h"
#include "tensorstore/serialization/fwd.h"
#include "tensorstore/serialization/json_bindable.h"
#include "tensorstore/strided_layout.h"
#include "tensorstore/util/constant_vector.h"
#include "tensorstore/util/endian.h"
#include "tensorstore/util/extents.h"
#include "tensorstore/util/result.h"
#include "tensorstore/util/span.h"
#include "tensorstore/util/status.h"
#include "tensorstore/util/str_cat.h"
#include "tensorstore/util/unit.h"
#include <iostream>

namespace tensorstore {
namespace internal_hdf5 {

namespace jb = tensorstore::internal_json_binding;
using ::tensorstore::internal::MetadataMismatchError;

namespace {

// (Jie): Currently not used
// const internal::CodecSpecRegistration<HDF5CodecSpec> encoding_registration;

constexpr std::array kSupportedDataTypes{
    DataTypeId::uint8_t,   DataTypeId::uint16_t, DataTypeId::uint32_t,
    DataTypeId::uint64_t,  DataTypeId::int8_t,   DataTypeId::int16_t,
    DataTypeId::int32_t,   DataTypeId::int64_t,  DataTypeId::float32_t,
    DataTypeId::float64_t,
};

std::string GetSupportedDataTypes() {
  return absl::StrJoin(
      kSupportedDataTypes, ", ", [](std::string* out, DataTypeId id) {
        absl::StrAppend(out, kDataTypes[static_cast<int>(id)].name());
      });
}

absl::Status ValidateMetadata(HDF5Metadata& metadata) {
  std::cout << "HDF5=====ValidateMetadata(HDF5Metadata& metadata)" << std::endl;
  // Check if HDF5 has some limitation for the metadata
  return absl::OkStatus();
}

constexpr auto MetadataJsonBinder = [](auto maybe_optional) {
  return [=](auto is_loading, const auto& options, auto* obj, auto* j) {
    using T = absl::remove_cvref_t<decltype(*obj)>;
    DimensionIndex* rank = nullptr;
    if constexpr (is_loading) {
      rank = &obj->rank;
    }

    return jb::Object(
        jb::Member(
            "shape", // the shape of the dataset. (TO DO): check what is dimensionIndexedVector?
            jb::Projection(&T::shape, maybe_optional(jb::ShapeVector(rank)))),
        
        jb::Member(
            "chunk_shape", // the chunk shape.
            jb::Projection(&T::chunk_shape, maybe_optional(jb::ChunkShapeVector(rank)))),
        
        jb::Member(
            "compression",
            jb::Projection(&T::compressor)),

        jb::Member(
            "data_type",
            jb::Projection(&T::dtype, maybe_optional(jb::Validate(
                                          [](const auto& options, auto* obj) {
                                            return ValidateDataType(*obj);
                                          },
                                          jb::DataTypeJsonBinder))))
        )(is_loading, options, obj, j);
  };
};

}  // namespace

// Who Call this function?? Is it required??
std::string HDF5Metadata::GetCompatibilityKey() const {
  std::cout << "HDF5=====GetCompatibilityKey()" << std::endl;
  ::nlohmann::json::object_t obj; 
  obj.emplace("chunk_shape", ::nlohmann::json::array_t(chunk_shape.begin(),
                                                     chunk_shape.end()));
  obj.emplace("data_type", dtype.name());
  obj.emplace("compression", compressor);
  
  return ::nlohmann::json(obj).dump();
}

TENSORSTORE_DEFINE_JSON_DEFAULT_BINDER(
    HDF5Metadata, jb::Validate([](const auto& options,
                                auto* obj) { return ValidateMetadata(*obj); },
                             MetadataJsonBinder(internal::identity{})))

TENSORSTORE_DEFINE_JSON_DEFAULT_BINDER(HDF5MetadataConstraints,
                                       MetadataJsonBinder([](auto binder) {
                                         return jb::Optional(binder);
                                       }))

///////////////////////////////////////////////////////////////////////////
// HDF5CodecSpec
CodecSpec HDF5CodecSpec::Clone() const {
  return internal::CodecDriverSpec::Make<HDF5CodecSpec>(*this);
}

absl::Status HDF5CodecSpec::DoMergeFrom(const internal::CodecDriverSpec& other_base) {
  if (typeid(other_base) != typeid(HDF5CodecSpec)) {
    return absl::InvalidArgumentError("CodecSpec types do not match");
  }
  auto& other = static_cast<const HDF5CodecSpec&>(other_base);
  if (other.compressor) {
    if (!compressor) {
      compressor = other.compressor;
    } else if (!internal_json::JsonSame(::nlohmann::json(*compressor),
                                        ::nlohmann::json(*other.compressor))) {
      return absl::InvalidArgumentError("HDF5 \"compression\" does not match");
    }
  }
  return absl::OkStatus();
}


TENSORSTORE_DEFINE_JSON_DEFAULT_BINDER(
    HDF5CodecSpec,
    jb::Sequence(jb::Member("compression",
                            jb::Projection(&HDF5CodecSpec::compressor))))
///////////////////////////////////////////////////////////////////////////

absl::Status ValidateMetadata(const HDF5Metadata& metadata,
                              const HDF5MetadataConstraints& constraints) {
  std::cout << "HDF5=====ValidateMetadata(const HDF5Metadata& metadata, const HDF5MetadataConstraints& constraints)" << std::endl;
  if (constraints.shape && !absl::c_equal(metadata.shape, *constraints.shape)) {
    return MetadataMismatchError("shape", *constraints.shape,
                                 metadata.shape);
  }
  if (constraints.chunk_shape &&
      !absl::c_equal(metadata.chunk_shape, *constraints.chunk_shape)) {
    return MetadataMismatchError("chunk_shape", *constraints.chunk_shape,
                                 metadata.chunk_shape);
  }
  if (constraints.dtype && *constraints.dtype != metadata.dtype) {
    return MetadataMismatchError("data_type", constraints.dtype->name(),
                                 metadata.dtype.name());
  }

  std::cout << "hhahahahah: " << ::nlohmann::json(metadata.compressor) << std::endl;
  if (constraints.compressor && ::nlohmann::json(*constraints.compressor) !=
                                    ::nlohmann::json(metadata.compressor)) {
    return MetadataMismatchError("compression", *constraints.compressor,
                                 metadata.compressor);
  }

  return absl::OkStatus();
}

Result<IndexDomain<>> GetEffectiveDomain(
    DimensionIndex rank, std::optional<span<const Index>> shape,
    const Schema& schema) {

  std::cout << "HDF5=====GetEffectiveDomain(DimensionIndex rank, std::optional<span<const Index>> shape, const Schema& schema)" << std::endl;
  auto domain = schema.domain();
  if (!shape && !domain.valid()) {
    if (schema.rank() == 0) return {std::in_place, 0};
    // No information about the domain available.
    return {std::in_place};
  }

  // Rank is already validated by caller.
  assert(RankConstraint::EqualOrUnspecified(schema.rank(), rank));
  IndexDomainBuilder builder(std::max(schema.rank().rank, rank));
  if (shape) {
    builder.shape(*shape);
    builder.implicit_upper_bounds(true);
  } else {
    builder.origin(GetConstantVector<Index, 0>(builder.rank()));
  }

  TENSORSTORE_ASSIGN_OR_RETURN(auto domain_from_metadata, builder.Finalize());
  TENSORSTORE_ASSIGN_OR_RETURN(domain,
                               MergeIndexDomains(domain, domain_from_metadata),
                               tensorstore::MaybeAnnotateStatus(
                                   _, "Mismatch between metadata and schema"));
  return WithImplicitDimensions(domain, false, true);
}

Result<IndexDomain<>> GetEffectiveDomain(
    const HDF5MetadataConstraints& metadata_constraints, const Schema& schema) {
  // This may has some problem, since I only have the shape.
  std::cout << "HDF5=====GetEffectiveDomain(const HDF5MetadataConstraints& metadata_constraints, const Schema& schema)" << std::endl;
  return GetEffectiveDomain(metadata_constraints.rank,
                            metadata_constraints.shape,
                            schema);
}

Result<std::shared_ptr<const HDF5Metadata>> GetNewMetadata(
    const HDF5MetadataConstraints& metadata_constraints, const Schema& schema) {
  
  std::cout << "HDF5=====GetNewMetadata()" << std::endl;
  auto metadata = std::make_shared<HDF5Metadata>();

  // Set domain
  TENSORSTORE_ASSIGN_OR_RETURN(
      auto domain, GetEffectiveDomain(metadata_constraints, schema));
  if (!domain.valid() || !IsFinite(domain.box())) {
    return absl::InvalidArgumentError("domain must be specified");
  }

  const DimensionIndex rank = metadata->rank = domain.rank();
  metadata->shape.assign(domain.shape().begin(), domain.shape().end());

  // Set dtype
  auto dtype = schema.dtype();
  if (!dtype.valid()) {
    return absl::InvalidArgumentError("dtype must be specified");
  }
  TENSORSTORE_RETURN_IF_ERROR(ValidateDataType(dtype));
  metadata->dtype = dtype;

  // (TODO) Set chunk shape (may be later check if HDF5 needed)
  
  // Set compressor (may be later check if HDF5 needed this)
  TENSORSTORE_ASSIGN_OR_RETURN(auto codec_spec,
                               GetEffectiveCodec(metadata_constraints, schema));
  if (codec_spec->compressor) {
    metadata->compressor = *codec_spec->compressor;
  }

  TENSORSTORE_RETURN_IF_ERROR(ValidateMetadata(*metadata));
  TENSORSTORE_RETURN_IF_ERROR(ValidateMetadataSchema(*metadata, schema));
  return metadata;
}

absl::Status ValidateMetadataSchema(const HDF5Metadata& metadata,
                                    const Schema& schema) {
  
  std::cout << "HDF5=====ValidateMetadataSchema(const HDF5Metadata& metadata, const Schema& schema)" << std::endl;
  if (!RankConstraint::EqualOrUnspecified(metadata.rank, schema.rank())) {
    return absl::FailedPreconditionError(tensorstore::StrCat(
        "Rank specified by schema (", schema.rank(),
        ") does not match rank specified by metadata (", metadata.rank, ")"));
  }

  if (schema.domain().valid()) { 
    TENSORSTORE_RETURN_IF_ERROR(GetEffectiveDomain(
        metadata.rank, metadata.shape, schema));
  }

  if (auto dtype = schema.dtype();
      !IsPossiblySameDataType(metadata.dtype, dtype)) {
    return absl::FailedPreconditionError(
        tensorstore::StrCat("dtype from metadata (", metadata.dtype,
                            ") does not match dtype in schema (", dtype, ")"));
  }

  if (auto schema_codec = schema.codec(); schema_codec.valid()) {
    auto codec = GetCodecFromMetadata(metadata);
    TENSORSTORE_RETURN_IF_ERROR(
        codec.MergeFrom(schema_codec),
        internal::ConvertInvalidArgumentToFailedPrecondition(
            tensorstore::MaybeAnnotateStatus(
                _, "codec from metadata does not match codec in schema")));
  }

  if (schema.chunk_layout().rank() != dynamic_rank) {
    // TENSORSTORE_ASSIGN_OR_RETURN(
    //     auto chunk_layout,
    //     GetEffectiveChunkLayout(metadata.rank, metadata.chunk_shape, schema));
    // if (chunk_layout.codec_chunk_shape().hard_constraint) {
    //   return absl::InvalidArgumentError("codec_chunk_shape not supported");
    // }
    std::cout << "Warning: schema.chunk_layout().rank() != dynamic_rank" << std::endl;
  }

  if (schema.fill_value().valid()) {
    std::cout << "Warning: schema.fill_value().valid() not supported for hdf5...." << std::endl;
    return absl::InvalidArgumentError("fill_value not supported by HDF5 format");
  }

  if (auto schema_units = schema.dimension_units(); schema_units.valid()) {
    // auto dimension_units =
    //     GetDimensionUnits(metadata.rank, metadata.units_and_resolution);
    // DimensionUnitsVector schema_units_vector(schema_units);
    // TENSORSTORE_RETURN_IF_ERROR(
    //     MergeDimensionUnits(schema_units_vector, dimension_units),
    //     internal::ConvertInvalidArgumentToFailedPrecondition(_));
    // if (schema_units_vector != dimension_units) {
    //   return absl::FailedPreconditionError(
    //       tensorstore::StrCat("Dimension units in metadata ",
    //                           DimensionUnitsToString(dimension_units),
    //                           " do not match dimension units in schema ",
    //                           DimensionUnitsToString(schema_units)));
    // }
    std::cout << "Warning: schema_units not supported for hdf5...." << std::endl;
  }
  return absl::OkStatus();
}

absl::Status ValidateDataType(DataType dtype) {
  if (!absl::c_linear_search(kSupportedDataTypes, dtype.id())) {
    return absl::InvalidArgumentError(tensorstore::StrCat(
        dtype, " data type is not one of the supported data types: ",
        GetSupportedDataTypes()));
  }
  return absl::OkStatus();
}

Result<internal::CodecDriverSpec::PtrT<HDF5CodecSpec>> GetEffectiveCodec(
    const HDF5MetadataConstraints& metadata_constraints, const Schema& schema) {
  std::cout << "HDF5=====GetEffectiveCodec(const HDF5MetadataConstraints& metadata_constraints, const Schema& schema)" << std::endl;
  auto codec_spec = internal::CodecDriverSpec::Make<HDF5CodecSpec>();
  if (metadata_constraints.compressor) {
    codec_spec->compressor = *metadata_constraints.compressor;
  }
  TENSORSTORE_RETURN_IF_ERROR(codec_spec->MergeFrom(schema.codec()));
  return codec_spec;

}

/// Returns the codec from the specified metadata.
CodecSpec GetCodecFromMetadata(const HDF5Metadata& metadata) {
  std::cout << "HDF5=====GetCodecFromMetadata(const HDF5Metadata& metadata)" << std::endl;
  auto codec_spec = internal::CodecDriverSpec::Make<HDF5CodecSpec>();
  codec_spec->compressor = metadata.compressor;
  return CodecSpec(std::move(codec_spec));
}

}  // namespace internal_hdf5
}  // namespace tensorstore

TENSORSTORE_DEFINE_SERIALIZER_SPECIALIZATION(
    tensorstore::internal_hdf5::HDF5MetadataConstraints,
    tensorstore::serialization::JsonBindableSerializer<
        tensorstore::internal_hdf5::HDF5MetadataConstraints>())
