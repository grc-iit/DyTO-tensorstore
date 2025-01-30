# Copyright 2020 The TensorStore Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# buildifier: disable=module-docstring

load("@bazel_tools//tools/build_defs/repo:utils.bzl", "maybe")
load("//third_party:repo.bzl", "third_party_http_archive")


def repo():
    maybe(
        third_party_http_archive,
        name = "rocksdb",
        strip_prefix = "rocksdb-8.0.0",
        urls = [
            "https://github.com/facebook/rocksdb/archive/refs/tags/v8.0.0.tar.gz",
        ],
        sha256 = "05ff6b0e89bffdf78b5a9d6fca46cb06bde6189f5787b9eeaef0511b782c1033",
        build_file = Label("//third_party:rocksdb/rocksdb.BUILD.bazel"),
        system_build_file = Label("//third_party:rocksdb/system.BUILD.bazel"),
    )

