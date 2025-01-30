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
#include <cassert>
#include <cmath>
#include <limits>
#include <memory>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

#include "absl/base/optimization.h"
#include "absl/container/inlined_vector.h"
#include "absl/meta/type_traits.h"
#include "absl/status/status.h"
#include "absl/strings/cord.h"
#include "absl/strings/escaping.h"
#include "absl/strings/match.h"
#include "absl/strings/numbers.h"
#include <nlohmann/json_fwd.hpp>
#include "riegeli/bytes/cord_reader.h"
#include "riegeli/bytes/cord_writer.h"
#include "riegeli/bytes/read_all.h"
#include "riegeli/bytes/reader.h"
#include "riegeli/bytes/write.h"
#include "riegeli/bytes/writer.h"
#include "tensorstore/array.h"
#include "tensorstore/contiguous_layout.h"
#include "tensorstore/data_type.h"
#include "tensorstore/index.h"
#include "tensorstore/internal/data_type_endian_conversion.h"
#include "tensorstore/internal/flat_cord_builder.h"
#include "tensorstore/internal/integer_overflow.h"
#include "tensorstore/internal/json/json.h"
#include "tensorstore/internal/json/value_as.h"
#include "tensorstore/internal/json_binding/bindable.h"
#include "tensorstore/internal/json_binding/data_type.h"
#include "tensorstore/internal/json_binding/dimension_indexed.h"
#include "tensorstore/internal/json_binding/enum.h"
#include "tensorstore/internal/json_binding/json_binding.h"
#include "tensorstore/internal/json_binding/std_optional.h"
#include "tensorstore/internal/riegeli/array_endian_codec.h"
#include "tensorstore/internal/type_traits.h"
#include "tensorstore/json_serialization_options_base.h"
#include "tensorstore/rank.h"
#include "tensorstore/serialization/fwd.h"
#include "tensorstore/serialization/json_bindable.h"
#include "tensorstore/strided_layout.h"
#include "tensorstore/util/byte_strided_pointer.h"
#include "tensorstore/util/element_pointer.h"
#include "tensorstore/util/endian.h"
#include "tensorstore/util/extents.h"
#include "tensorstore/util/result.h"
#include "tensorstore/util/span.h"
#include "tensorstore/util/status.h"
#include "tensorstore/util/str_cat.h"

namespace tensorstore {
namespace internal_hdf5 {

namespace jb = internal_json_binding;

// (TODO): Check how to change this to HDF5 metadata
constexpr auto MetadataJsonBinder = [](auto maybe_optional) {
  return [=](auto is_loading, const auto& options, auto* obj, auto* j) {
    using T = absl::remove_cvref_t<decltype(*obj)>;
    DimensionIndex* rank = nullptr;
    if constexpr (is_loading) {
      rank = &obj->rank;
    }

    return jb::Object(
        jb::Member(
            "shape",
            jb::Projection(&T::shape, maybe_optional(jb::ShapeVector(rank)))),
        jb::Member("chunks",
                   jb::Projection(&T::chunks,
                                  maybe_optional(jb::ChunkShapeVector(rank)))),
        jb::Member(
            "dataType",
            jb::Projection(&T::dtype, maybe_optional(jb::Validate(
                                          [](const auto& options, auto* obj) {
                                            return ValidateDataType(*obj);
                                          },
                                          jb::DataTypeJsonBinder)))),
        // jb::Member("compressor", jb::Projection(&T::compressor)),
        jb::Member("fill_value",
                   jb::Projection(
                       &T::fill_value,
                       maybe_optional([&](auto is_loading, const auto& options,
                                          auto* obj, auto* j) {
                         return FillValueJsonBinder(*dtype)(is_loading, options,
                                                            obj, j);
                       }))),
        // jb::Member("order",
        //            jb::Projection(&T::order, maybe_optional(OrderJsonBinder))),
        jb::Member(
            "filters",
            jb::Projection<&T::filters>(maybe_optional(
                [](auto is_loading, const auto& options, auto* obj, auto* j) {
                  if constexpr (is_loading) {
                    if (!j->is_null()) {
                      if (auto* a = j->template get_ptr<
                                    const ::nlohmann::json::array_t*>()) {
                        if (a->size() != 0) {
                          return absl::InvalidArgumentError(
                              "filters are not supported");
                        }
                      } else {
                        return internal_json::ExpectedError(*j,
                                                            "null or array");
                      }
                    }
                  } else {
                    *j = nullptr;
                  }
                  return absl::OkStatus();
                }))),
        // jb::Member("dimension_separator",
        //            jb::Projection(&T::dimension_separator,
        //                           jb::Optional(DimensionSeparatorJsonBinder))),
        [](auto is_loading, const auto& options, auto* obj, auto* j) {
          if constexpr (std::is_same_v<T, HDF5Metadata>) {
            return jb::DefaultBinder<>(is_loading, options, &obj->extra_members,
                                       j);
          } else {
            return absl::OkStatus();
          }
        })(is_loading, options, obj, j);
  };
};

absl::Status ValidateMetadata(HDF5Metadata& metadata) {
  // TENSORSTORE_ASSIGN_OR_RETURN(
  //     metadata.chunk_layout,
  //     ComputeChunkLayout(metadata.dtype, metadata.order, metadata.chunks));
  return absl::OkStatus();
}

TENSORSTORE_DEFINE_JSON_DEFAULT_BINDER(
    HDF5Metadata, jb::Validate([](const auto& options,
                                  auto* obj) { return ValidateMetadata(*obj); },
                               MetadataJsonBinder(internal::identity{})))

TENSORSTORE_DEFINE_JSON_DEFAULT_BINDER(HDF5PartialMetadata,
                                       MetadataJsonBinder([](auto binder) {
                                         return jb::Optional(binder);
                                       }))

void EncodeCacheKeyAdl(std::string* out, const HDF5Metadata& metadata) {
  auto json = ::nlohmann::json(metadata);
  json["shape"] = metadata.shape.size();
  out->append(json.dump());
}

}  // namespace internal_hdf5
}  // namespace tensorstore

TENSORSTORE_DEFINE_SERIALIZER_SPECIALIZATION(
    tensorstore::internal_hdf5::HDF5PartialMetadata,
    tensorstore::serialization::JsonBindableSerializer<
        tensorstore::internal_hdf5::HDF5PartialMetadata>())
