// Copyright 2023 The TensorStore Authors
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

#include "tensorstore/driver/driver.h"

#include <stddef.h>

#include <algorithm>
#include <cassert>
#include <memory>
#include <string>
#include <string_view>
#include <utility>

#include "absl/status/status.h"
#include "absl/strings/ascii.h"
#include "absl/strings/cord.h"
#include "absl/strings/numbers.h"
#include "absl/strings/str_cat.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include <nlohmann/json.hpp>
#include "tensorstore/array.h"
#include "tensorstore/array_storage_statistics.h"
#include "tensorstore/box.h"
#include "tensorstore/chunk_layout.h"
#include "tensorstore/codec_spec.h"
#include "tensorstore/contiguous_layout.h"
#include "tensorstore/data_type.h"
#include "tensorstore/driver/chunk.h"
#include "tensorstore/driver/driver_spec.h"
#include "tensorstore/driver/kvs_backed_chunk_driver.h"
#include "tensorstore/driver/registry.h"
// #include "tensorstore/driver/zarr3/chunk_cache.h"
#include "tensorstore/driver/hdf5/metadata.h"
#include "tensorstore/index.h"
#include "tensorstore/index_interval.h"
#include "tensorstore/index_space/dimension_units.h"
#include "tensorstore/index_space/index_domain.h"
#include "tensorstore/index_space/index_transform.h"
#include "tensorstore/index_space/index_transform_builder.h"
#include "tensorstore/internal/async_write_array.h"
#include "tensorstore/internal/cache/cache.h"
#include "tensorstore/internal/cache_key/cache_key.h"
#include "tensorstore/internal/chunk_grid_specification.h"
#include "tensorstore/internal/grid_chunk_key_ranges_base10.h"
#include "tensorstore/internal/intrusive_ptr.h"
#include "tensorstore/internal/json_binding/json_binding.h"
#include "tensorstore/internal/lexicographical_grid_index_key.h"
#include "tensorstore/internal/storage_statistics.h"
#include "tensorstore/open_mode.h"
#include "tensorstore/open_options.h"
#include "tensorstore/rank.h"
#include "tensorstore/transaction.h"
#include "tensorstore/util/dimension_set.h"
#include "tensorstore/util/execution/any_receiver.h"
#include "tensorstore/util/executor.h"
#include "tensorstore/util/future.h"
#include "tensorstore/util/garbage_collection/fwd.h"
#include "tensorstore/util/result.h"
#include "tensorstore/util/span.h"
#include "tensorstore/util/status.h"
#include "tensorstore/util/str_cat.h"
#include <iostream>

namespace tensorstore {
namespace internal_hdf5 {

// Avoid anonymous namespace to workaround MSVC bug.
//
// https://developercommunity.visualstudio.com/t/Bug-involving-virtual-functions-templat/10424129
#ifndef _MSC_VER
namespace {
#endif

namespace jb = tensorstore::internal_json_binding;

using ::tensorstore::internal_kvs_backed_chunk_driver::KvsDriverSpec;

constexpr const char kMetadataKey[] = "h5_meta.json";

class HDF5DriverSpec
    : public internal::RegisteredDriverSpec<HDF5DriverSpec,
                                            /*Parent=*/KvsDriverSpec> {
 public:
  constexpr static char id[] = "hdf5";

  using Base = internal::RegisteredDriverSpec<HDF5DriverSpec,
                                              /*Parent=*/KvsDriverSpec>;

  HDF5MetadataConstraints metadata_constraints;

  constexpr static auto ApplyMembers = [](auto& x, auto f) {
    std::cout << "HDF5DriverSpec::ApplyMembers --- x.metdata_constraints: "
    << x.metadata_constraints.ToJson().value() << std::endl;
    return f(internal::BaseCast<KvsDriverSpec>(x), x.metadata_constraints);
  };

  static inline const auto default_json_binder = jb::Sequence(
      jb::Validate(
          [](const auto& options, auto* obj) {
            if (obj->schema.dtype().valid()) {
              std::cout << "HDF5 default_json_binder's: " << (tensorstore::StrCat(obj->schema.dtype(), " is current dtype")) << std::endl;
              return ValidateDataType(obj->schema.dtype());
            }
            return absl::OkStatus();
          },
          internal_kvs_backed_chunk_driver::SpecJsonBinder),
      jb::Member(
          "metadata",
          jb::Validate(
              [](const auto& options, auto* obj) {
                TENSORSTORE_RETURN_IF_ERROR(obj->schema.Set(
                    obj->metadata_constraints.dtype.value_or(DataType())));
                TENSORSTORE_RETURN_IF_ERROR(obj->schema.Set(
                    RankConstraint{obj->metadata_constraints.rank}));
                return absl::OkStatus();
              },
              jb::Projection<&HDF5DriverSpec::metadata_constraints>(
                  jb::DefaultInitializedValue())))
      );

  absl::Status ApplyOptions(SpecOptions&& options) override {
    std::cout << "HDF5DriverSpec::ApplyOptions options.minimal_spec = " << options.minimal_spec << std::endl;
    if (options.minimal_spec) {
      metadata_constraints = HDF5MetadataConstraints{};
    }
    return Base::ApplyOptions(std::move(options));
  }

  Result<IndexDomain<>> GetDomain() const override {
    std::cout << "HDF5DriverSpec::GetDomain()" << std::endl;
    return GetEffectiveDomain(metadata_constraints, schema);
  }

  Result<CodecSpec> GetCodec() const override {
    std::cout << "HDF5DriverSpec::GetCodec()" << std::endl;
    return Base::GetCodec();
  }

  Result<ChunkLayout> GetChunkLayout() const override {
    std::cout << "HDF5DriverSpec::GetChunkLayout()" << std::endl;
    return Base::GetChunkLayout();
  }

  Future<internal::Driver::Handle> Open(
      internal::DriverOpenRequest request) const override;
};

// Result<std::shared_ptr<const ZarrMetadata>> ParseEncodedMetadata(
//     std::string_view encoded_value) {
//   nlohmann::json raw_data = nlohmann::json::parse(encoded_value, nullptr,
//                                                   /*allow_exceptions=*/false);
//   if (raw_data.is_discarded()) {
//     return absl::DataLossError("Invalid JSON");
//   }
//   TENSORSTORE_ASSIGN_OR_RETURN(auto metadata,
//                                ZarrMetadata::FromJson(std::move(raw_data)));
//   return std::make_shared<ZarrMetadata>(std::move(metadata));
// }

class MetadataCache : public internal_kvs_backed_chunk_driver::MetadataCache {
  using Base = internal_kvs_backed_chunk_driver::MetadataCache;

 public:
  using Base::Base;

  // TODO: Think how to do this for HDF5
  //  Metadata is stored as JSON under the `hdf5_meta.json` key.
  std::string GetMetadataStorageKey(std::string_view entry_key) override {
    std::cout << "HDF5 MetadataCache::GetMetadataStorageKey(): " << entry_key << "+" 
      << kMetadataKey << std::endl;
    return tensorstore::StrCat(entry_key, kMetadataKey);
  }

  Result<MetadataPtr> DecodeMetadata(std::string_view entry_key,
                                     absl::Cord encoded_metadata) override {
    std::cout << "HDF5 MetadataCache::DecodeMetadata() with key (" << entry_key 
      << ")" << std::endl;
    // return ParseEncodedMetadata(encoded_metadata.Flatten());
    return std::make_shared<HDF5Metadata>();
  }

  Result<absl::Cord> EncodeMetadata(std::string_view entry_key,
                                    const void* metadata) override {
    std::cout << "HDF5 MetadataCache::EncodeMetadata() for key (" << entry_key 
      << ")" << std::endl;
    // return absl::Cord(
    //     ::nlohmann::json(*static_cast<const HDF5Metadata*>(metadata)).dump());
    return absl::Cord();
  }
};


class DataCache : public internal_kvs_backed_chunk_driver::DataCache {
  using Base = internal_kvs_backed_chunk_driver::DataCache;

 public:
  explicit DataCache(Initializer&& initializer, std::string key_prefix)
      : Base(std::move(initializer),
             GetChunkGridSpecification(
                 *static_cast<const HDF5Metadata*>(initializer.metadata.get()))),
        key_prefix_(std::move(key_prefix)) {
          std::cout << "HDF5 DataCache::DataCache()" << std::endl;
        }

  absl::Status ValidateMetadataCompatibility(
      const void* existing_metadata_ptr,
      const void* new_metadata_ptr) override {
    std::cout << "HDF5 DataCache::ValidateMetadataCompatibility()" << std::endl;
    const auto& existing_metadata =
        *static_cast<const HDF5Metadata*>(existing_metadata_ptr);
    const auto& new_metadata =
        *static_cast<const HDF5Metadata*>(new_metadata_ptr);
    auto existing_key = existing_metadata.GetCompatibilityKey();
    auto new_key = new_metadata.GetCompatibilityKey();
    if (existing_key == new_key) return absl::OkStatus();
    return absl::FailedPreconditionError(tensorstore::StrCat(
        "Updated HDF5 metadata ", new_key,
        " is incompatible with existing metadata ", existing_key));
  }

  void GetChunkGridBounds(const void* metadata_ptr, MutableBoxView<> bounds,
                          DimensionSet& implicit_lower_bounds,
                          DimensionSet& implicit_upper_bounds) override {
    // const auto& metadata = *static_cast<const HDF5Metadata*>(metadata_ptr);
    // assert(bounds.rank() == static_cast<DimensionIndex>(metadata.shape.size()));
    // std::fill(bounds.origin().begin(), bounds.origin().end(), Index(0));
    // std::copy(metadata.shape.begin(), metadata.shape.end(),
    //           bounds.shape().begin());
    // implicit_lower_bounds = false;
    // implicit_upper_bounds = true;
    std::cout << "*****************HDF5 DataCache::GetChunkGridBounds()" << std::endl;
  }

  Result<std::shared_ptr<const void>> GetResizedMetadata(
      const void* existing_metadata, span<const Index> new_inclusive_min,
      span<const Index> new_exclusive_max) override {
    std::cout << "HDF5 DataCache::GetResizedMetadata()" << std::endl;
    auto new_metadata = std::make_shared<HDF5Metadata>(
        *static_cast<const HDF5Metadata*>(existing_metadata));
    // const DimensionIndex rank = new_metadata->shape.size();
    // assert(rank == new_inclusive_min.size());
    // assert(rank == new_exclusive_max.size());
    // for (DimensionIndex i = 0; i < rank; ++i) {
    //   assert(ExplicitIndexOr(new_inclusive_min[i], 0) == 0);
    //   const Index new_size = new_exclusive_max[i];
    //   if (new_size == kImplicit) continue;
    //   new_metadata->shape[i] = new_size;
    // }
    return new_metadata;
  }

  static internal::ChunkGridSpecification GetChunkGridSpecification(
      const HDF5Metadata& metadata) {
    std::cout << "*****************HDF5 DataCache::GetChunkGridSpecification()" << std::endl;
    std::cout << "HDF5 DataCache::GetChunkGridSpecification() metadata:\n"
    << metadata.ToJson().value() << metadata.rank << std::endl;
    // auto fill_value = BroadcastArray(AllocateArray(
    //                                      /*shape=*/span<const Index>{}, c_order,
    //                                      value_init, metadata.dtype),
    //                                  BoxView<>(metadata.rank))
    //                       .value();
    internal::ChunkGridSpecification::ComponentList components;
    // components.emplace_back(
    //     internal::AsyncWriteArray::Spec{
    //         std::move(fill_value),
    //         // Since all dimensions are resizable, just specify
    //         // unbounded `component_bounds`.
    //         Box<>(metadata.rank), fortran_order},
    //     metadata.chunk_shape);
    return internal::ChunkGridSpecification(std::move(components));
  }

  const HDF5Metadata& metadata() {
    std::cout << "*****************HDF5 DataCache::metadata()" << std::endl;
    return *static_cast<const HDF5Metadata*>(initial_metadata().get());
  }

  Result<absl::InlinedVector<SharedArray<const void>, 1>> DecodeChunk(
      span<const Index> chunk_indices, absl::Cord data) override {
    std::cout << "*****************HDF5 DataCache::DecodeChunk()" << std::endl;
    // TENSORSTORE_ASSIGN_OR_RETURN(
    //     auto array, internal_hdf5::DecodeChunk(metadata(), std::move(data)));
    // absl::InlinedVector<SharedArray<const void>, 1> components;
    // components.emplace_back(std::move(array));
    // return components;
    // return Base::DecodeChunk(chunk_indices, data);
    return absl::InlinedVector<SharedArray<const void>, 1>();
  }

  Result<absl::Cord> EncodeChunk(
      span<const Index> chunk_indices,
      span<const SharedArray<const void>> component_arrays) override {
    // assert(component_arrays.size() == 1);
    // return internal_hdf5::EncodeChunk(metadata(), component_arrays[0]);
    std::cout << "*****************HDF5 DataCache::EncodeChunk()" << std::endl;
    // return Base::EncodeChunk(chunk_indices, component_arrays);
    return absl::Cord();
  }

  std::string GetChunkStorageKey(span<const Index> cell_indices) override {
    // // Use "0" for rank 0 as a special case.
    // std::string key = tensorstore::StrCat(
    //     key_prefix_, cell_indices.empty() ? 0 : cell_indices[0]);
    // for (DimensionIndex i = 1; i < cell_indices.size(); ++i) {
    //   tensorstore::StrAppend(&key, "/", cell_indices[i]);
    // }
    // return key;
    std::cout << "*****************HDF5 DataCache::GetChunkStorageKey()" << std::endl;
    return "xxxxx";
  }

  Result<IndexTransform<>> GetExternalToInternalTransform(
      const void* metadata_ptr, size_t component_index) override {
    std::cout << "*****************HDF5 DataCache::GetExternalToInternalTransform()" << std::endl;
    return Base::GetExternalToInternalTransform(metadata_ptr, component_index);
  }

  absl::Status GetBoundSpecData(KvsDriverSpec& spec_base,
                                const void* metadata_ptr,
                                size_t component_index) override {
    std::cout << "*****************HDF5 DataCache::GetBoundSpecData()" << std::endl;
    assert(component_index == 0);
    auto& spec = static_cast<HDF5DriverSpec&>(spec_base);
    const auto& metadata = *static_cast<const HDF5Metadata*>(metadata_ptr);
    auto& constraints = spec.metadata_constraints;
    constraints.shape = metadata.shape;
    constraints.dtype = metadata.dtype;
    constraints.compressor = metadata.compressor;
    constraints.chunk_shape = metadata.chunk_shape;
    return absl::OkStatus();
  }

  Result<ChunkLayout> GetChunkLayoutFromMetadata(
      const void* metadata_ptr, size_t component_index) override {
    std::cout << "*****************HDF5 DataCache::GetChunkLayoutFromMetadata()" << std::endl;
    const auto& metadata = *static_cast<const HDF5Metadata*>(metadata_ptr);
    ChunkLayout chunk_layout;
    // TENSORSTORE_RETURN_IF_ERROR(SetChunkLayoutFromMetadata(
    //     metadata.rank, metadata.chunk_shape, chunk_layout));
    // TENSORSTORE_RETURN_IF_ERROR(chunk_layout.Finalize());
    return chunk_layout;
  }

  std::string GetBaseKvstorePath() override { return key_prefix_; }

  std::string key_prefix_;
};


class HDF5Driver;
using HDF5DriverBase = internal_kvs_backed_chunk_driver::RegisteredKvsDriver<
    HDF5Driver, HDF5DriverSpec, DataCache,
    internal_kvs_backed_chunk_driver::KvsChunkedDriverBase>;

class HDF5Driver : public HDF5DriverBase {
  using Base = HDF5DriverBase;

 public:
  using Base::Base;
  const HDF5Metadata& metadata() const {
    std::cout << "*****HDF5Driver::metadata()" << std::endl;
    return *static_cast<const HDF5Metadata*>(cache()->initial_metadata().get());
  }

  Future<ArrayStorageStatistics> GetStorageStatistics(
      GetStorageStatisticsRequest request) override;

  Result<CodecSpec> GetCodec() override {
    std::cout << "*******HDF5Driver::GetCodec()" << std::endl;
    return GetCodecFromMetadata(metadata());
  }

  void Read(ReadRequest request, 
          AnyFlowReceiver<absl::Status, internal::ReadChunk, IndexTransform<>> 
          receiver) override {
    // return cache()->zarr_chunk_cache().Read(
    //     {std::move(request), GetCurrentDataStalenessBound(),
    //      this->fill_value_mode_.fill_missing_data_reads},
    //     std::move(receiver));
    std::cout << "****************HDF5Driver::Read" << std::endl;
    return;
  }

  void Write(
      WriteRequest request,
      AnyFlowReceiver<absl::Status, internal::WriteChunk, IndexTransform<>>
          receiver) override {
    // return cache()->zarr_chunk_cache().Write(
    //     {std::move(request),
    //      this->fill_value_mode_.store_data_equal_to_fill_value},
    //     std::move(receiver));
    std::cout << "****************HDF5Driver::Write" << std::endl;
    return;
  }

  // absl::Time GetCurrentDataStalenessBound() {
  //   absl::Time bound = this->data_staleness_bound().time;
  //   if (bound != absl::InfinitePast()) {
  //     bound = std::min(bound, absl::Now());
  //   }
  //   return bound;
  // }

  class OpenState;
};

Future<ArrayStorageStatistics> HDF5Driver::GetStorageStatistics(
    GetStorageStatisticsRequest request) {
  std::cout << "****************HDF5Driver::GetStorageStatistics" << std::endl;
  Future<ArrayStorageStatistics> future;
  // Note: `future` is an output parameter.
  auto state = internal::MakeIntrusivePtr<
      internal::GetStorageStatisticsAsyncOperationState>(future,
                                                         request.options);
  // auto* state_ptr = state.get();
  // auto* cache = this->cache();
  // auto transaction = request.transaction;
  // LinkValue(
  //     WithExecutor(cache->executor(),
  //                  [state = std::move(state),
  //                   cache = internal::CachePtr<DataCacheBase>(cache),
  //                   transform = std::move(request.transform),
  //                   transaction = std::move(request.transaction),
  //                   staleness_bound = this->GetCurrentDataStalenessBound()](
  //                      Promise<ArrayStorageStatistics> promise,
  //                      ReadyFuture<MetadataCache::MetadataPtr> future) mutable {
  //                    auto* metadata =
  //                        static_cast<const HDF5Metadata*>(future.value().get());
  //                    cache->zarr_chunk_cache().GetStorageStatistics(
  //                        std::move(state),
  //                        {std::move(transaction), metadata->shape,
  //                         std::move(transform), staleness_bound});
  //                  }),
  //     state_ptr->promise,
  //     ResolveMetadata(std::move(transaction), metadata_staleness_bound_.time));
  return future;
}

//////////////////////////////////////////////////////////////////////////////
class HDF5Driver::OpenState : public HDF5Driver::OpenStateBase {
 public:
  using HDF5Driver::OpenStateBase::OpenStateBase;

  std::string GetPrefixForDeleteExisting() override {
    std::cout << "****************HDF5Driver::OpenState::GetPrefixForDeleteExisting" << std::endl;
    return spec().store.path;
  }

  std::string GetMetadataCacheEntryKey() override { 
    std::cout << "****************HDF5Driver::OpenState::GetMetadataCacheEntryKey\n\t" 
      << spec().store.path << std::endl;
    return spec().store.path; 
  }

  // The metadata cache isn't parameterized by anything other than the
  // KeyValueStore; therefore, we don't need to override `GetMetadataCacheKey`
  // to encode the state.
  std::unique_ptr<internal_kvs_backed_chunk_driver::MetadataCache>
  GetMetadataCache(MetadataCache::Initializer initializer) override {
    std::cout << "****************HDF5Driver::OpenState::GetMetadataCache" << std::endl;
    return std::make_unique<MetadataCache>(std::move(initializer));
  }

  std::string GetDataCacheKey(const void* metadata) override {
    std::cout << "****************HDF5Driver::OpenState::GetDataCacheKey" << std::endl;
    std::string result = "teststtststs-DataCacheKey";
    // internal::EncodeCacheKey(
    //     &result, spec().store.path,
    //     static_cast<const HDF5Metadata*>(metadata)->GetCompatibilityKey());
    return result;
  }

  Result<std::shared_ptr<const void>> Create(const void* existing_metadata,
                                             CreateOptions options) override {
    std::cout << "****************HDF5Driver::OpenState::Create" << std::endl;
    if (existing_metadata) {
      return absl::AlreadyExistsError("The metadata already exists");
    }
    TENSORSTORE_ASSIGN_OR_RETURN(
        auto metadata,
        internal_hdf5::GetNewMetadata(spec().metadata_constraints, spec().schema),
        tensorstore::MaybeAnnotateStatus(
            _, "Cannot create using specified \"metadata\" and schema"));
    return metadata;
  }

  std::unique_ptr<internal_kvs_backed_chunk_driver::DataCacheBase> GetDataCache(
      DataCache::Initializer&& initializer) override {
    std::cout << "****************HDF5Driver::OpenState::GetDataCache" << std::endl;
    return std::make_unique<DataCache>(std::move(initializer),
                                       spec().store.path);
  }

  Result<size_t> GetComponentIndex(const void* metadata_ptr,
                                   OpenMode open_mode) override {
    std::cout << "****************HDF5Driver::OpenState::GetComponentIndex" << std::endl;
    // const auto& metadata = *static_cast<const HDF5Metadata*>(metadata_ptr);
    // TENSORSTORE_RETURN_IF_ERROR(
    //     ValidateMetadata(metadata, spec().metadata_constraints));
    // TENSORSTORE_RETURN_IF_ERROR(
    //     ValidateMetadataSchema(metadata, spec().schema));
    return 0;
  }
};

Future<internal::Driver::Handle> HDF5DriverSpec::Open(
    internal::DriverOpenRequest request) const {
      std::cout << "*********HDF5DriverSpec::Open, request = " << request.read_write_mode << std::endl;
  return HDF5Driver::Open(this, std::move(request));
}

#ifndef _MSC_VER
}  // namespace
#endif

}  // namespace internal_hdf5
}  // namespace tensorstore

TENSORSTORE_DECLARE_GARBAGE_COLLECTION_SPECIALIZATION(
    tensorstore::internal_hdf5::HDF5Driver)
// Use default garbage collection implementation provided by
// kvs_backed_chunk_driver (just handles the kvstore)
TENSORSTORE_DEFINE_GARBAGE_COLLECTION_SPECIALIZATION(
    tensorstore::internal_hdf5::HDF5Driver,
    tensorstore::internal_hdf5::HDF5Driver::GarbageCollectionBase)

namespace {
const tensorstore::internal::DriverRegistration<
    tensorstore::internal_hdf5::HDF5DriverSpec>
    registration;
}  // namespace
