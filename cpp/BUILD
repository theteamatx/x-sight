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

# Sight logging library.

load(
    "//google/api/build_defs:producer.bzl",
    "api_cli",
    "api_discovery",
    "api_discovery_validate",
    "api_doc",
    "api_publish",
    "api_service",
    "api_validation_test",
)
load("//javascript/typescript/contrib/apiclient:build_defs.bzl", "ts_api_client")
load("//apiserving/libgen/gen:def.bzl", "javascript_apiary_externs", "javascript_discovery_library")

package(default_visibility = [
    "//visibility:public",
])

cc_library(
    name = "sight",
    srcs = ["sight.cc"],
    hdrs = ["sight.h"],
    deps = [
        ":sight_interface",
        "//analysis/dremel/core/capacitor/public:importer_options_cc_proto",
        "//analysis/dremel/core/capacitor/public:message-importer",
        "//analysis/dremel/core/capacitor/public:schema",
        "//base",
        "//base:timer",
        "//file/base",
        "//file/base:path",
        "//googlex/cortex/sight/proto:sight_cc_proto",
        "//googlex/cortex/sight/service/v1:sight_service_cc_grpc_proto",
        "//net/grpc",
        "//net/grpc:grpc++",
        "//repository/base:repository_cc_proto",
        "//segindexer:compositedoc_cc_proto",
        "//tech/env:envelope_loader",
        "//tech/env:envelope_optional",  # for google3 name resolution.
        "//third_party/absl/memory",
        "//third_party/absl/status",
        "//third_party/absl/status:statusor",
        "//third_party/absl/strings",
        "//third_party/grpc:grpc++",
        "//util/gtl:map_util",
        "//util/gtl:ptr_util",
        "//util/task:status",
    ],
)

cc_library(
    name = "sight_core_interface",
    srcs = ["sight_core_interface.cc"],
    hdrs = ["sight_core_interface.h"],
    deps = [
        "//base",
        "//file/base",
        "//file/base:path",
        "//googlex/cortex/sight/proto:sight_cc_proto",
        "//third_party/absl/memory",
        "//third_party/absl/strings",
        "//util/gtl:map_util",
        "//util/task:status",
    ],
)

cc_library(
    name = "sight_interface",
    hdrs = ["sight_interface.h"],
    deps = [
        ":sight_core_interface",
        "//base",
        "//base:timer",
        "//file/base",
        "//file/base:path",
        "//third_party/absl/strings",
        "//util/task:status",
    ],
)

cc_test(
    name = "sight_test",
    srcs = ["sight_test.cc"],
    deps = [
        ":sight",
        "//analysis/dremel/core/capacitor/public:record-reader",
        "//file/base",
        "//file/base:path",
        "//file/columnio/public:record-reader",
        "//testing/base/public:gunit_main",
    ],
)
