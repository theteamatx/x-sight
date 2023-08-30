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

# Definition of Sight service API.

load("//tools/build_defs/proto/cpp:cc_grpc_library.bzl", "cc_grpc_library")
load("//net/proto2/compiler/stubby/java:java_stubby_library.bzl", "java_stubby_library")
load("//net/grpc/python:py_grpc_library.bzl", "py_grpc_library")
load("//tools/build_defs/js/jspb_proto_library:rules.bzl", "jspb_proto_library")
load("//tools/build_defs/proto/cpp:cc_proto_library.bzl", "cc_proto_library")
load("//third_party/java/grpc:build_defs.bzl", "java_grpc_library")

package(default_visibility = ["//visibility:public"])

proto_library(
    name = "sight_service_proto",
    srcs = ["sight_service.proto"],
    has_services = 1,
    use_java_stubby_library = True,
    deps = [
        "//google/api:annotations",
        "//googlex/cortex/sight/proto:sight_proto",
        "//java/com/google/apps/jspb:jspb_proto",
    ],
    alwayslink = 1,
)

java_stubby_library(
    name = "sight_service_java_stubby",
    srcs = [":sight_service_proto"],
    deps = [":sight_service_java_proto"],
)

cc_proto_library(
    name = "sight_service_cc_proto",
    deps = [":sight_service_proto"],
)

cc_grpc_library(
    name = "sight_service_cc_grpc_proto",
    srcs = [":sight_service_proto"],
    deps = [":sight_service_cc_proto"],
)

go_proto_library(
    name = "sight_service_go_proto",
    deps = [":sight_service_proto"],
)

java_proto_library(
    name = "sight_service_java_proto",
    deps = [":sight_service_proto"],
)

jspb_proto_library(
    name = "sight_service_jspb_proto",
    deps = [":sight_service_proto"],
)

py_proto_library(
    name = "sight_service_py_pb2",
    has_services = 1,
    api_version = 2,
    deps = [":sight_service_proto"],
)

java_grpc_library(
    name = "sight_service_java_grpc",
    srcs = [":sight_service_proto"],
    deps = [":sight_service_java_proto"],
)

py_grpc_library(
    name = "sight_service_py_pb2_grpc",
    srcs = [":sight_service_proto"],
    deps = [":sight_service_py_pb2"],
)
