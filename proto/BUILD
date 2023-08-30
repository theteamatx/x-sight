# Copyright 2023 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Definitions of the protobuf encoding of Sight logs.

load("//net/proto2/util/public:proto2_db.bzl", "proto2_db")
load("//tools/build_defs/proto/cpp:cc_proto_library.bzl", "cc_proto_library")

package(default_visibility = ["//visibility:public"])

# Proto libraries
proto_library(
    name = "sight_proto",
    srcs = [
        "sight.proto",
    ],
    cc_api_version = 2,
    deps = [
        "//googlex/cortex/sight/widgets/pipeline/flume/proto:flume_proto",
        "//third_party/tensorflow/core/example:example_protos",
    ],
    alwayslink = 1,
)

cc_proto_library(
    name = "sight_cc_proto",
    deps = [":sight_proto"],
)

java_proto_library(
    name = "sight_java_proto",
    deps = [":sight_proto"],
)

py_proto_library(
    name = "sight_py_pb2",
    api_version = 2,
    deps = [":sight_proto"],
)

proto2_db(
    name = "sight_proto2db",
    srcs = [
        ":sight_proto",
    ],
)
