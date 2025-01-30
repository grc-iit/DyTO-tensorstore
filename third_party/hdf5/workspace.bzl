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
        name = "hdf5",
        strip_prefix = "hdf5-1.14.5",
        urls = [
            "https://github.com/HDFGroup/hdf5/releases/download/hdf5_1.14.5/hdf5-1.14.5.tar.gz",
        ],
        sha256 = "ec2e13c52e60f9a01491bb3158cb3778c985697131fc6a342262d32a26e58e44",
        build_file = Label("//third_party:hdf5/hdf5.BUILD.bazel"),
        system_build_file = Label("//third_party:hdf5/system.BUILD.bazel"),
    )

