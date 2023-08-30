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

# Sight logging framework

load(
    "//java/com/google/devtools/staticanalysis/java/checker:build_defs.bzl",
    "java_library_with_nullness_check",
)

package(default_visibility = [
    "//java/com/google/googlex/cortex/sight:__subpackages__",
    "//javatests/com/google/googlex/cortex/sight:__subpackages__",
])

java_library_with_nullness_check(
    name = "Sight",
    srcs = [
        "Attribute.java",
        "Block.java",
        "Collections.java",
        "Location.java",
        "Sight.java",
        "SightCore.java",
        "SightImpl.java",
        "SightLoggable.java",
        "widgets/flume/DoFn.java",
        "widgets/flume/DoFnProcessBlock.java",
        "widgets/flume/DoFnUtils.java",
        "widgets/flume/DoToTableFn.java",
        "widgets/flume/EmitFn.java",
        "widgets/flume/EmitToTableFn.java",
        "widgets/flume/FlumeUtils.java",
        "widgets/flume/MapFn.java",
        "widgets/flume/MapToTableFn.java",
        "widgets/flume/PCollection.java",
        "widgets/flume/PTable.java",
        "widgets/flume/SightId.java",
        "widgets/protobuf/Proto.java",
    ],
    deps = [
        "//analysis/dremel/core/capacitor/public:importer-options_java_proto",
        "//googlex/cortex/sight/proto:sight_java_proto",
        "//googlex/cortex/sight/service/v1:sight_service_java_grpc",
        "//googlex/cortex/sight/service/v1:sight_service_java_proto",
        "//googlex/cortex/sight/widgets/pipeline/flume/proto:flume_java_proto",
        "//java/com/google/analysis/dremel/core/capacitor:common",
        "//java/com/google/analysis/dremel/core/capacitor:message-importer",
        "//java/com/google/borg/util",
        "//java/com/google/common/base",
        "//java/com/google/common/collect",
        "//java/com/google/common/primitives",
        "//java/com/google/common/reflect",
        "//java/com/google/common/time",
        "//java/com/google/common/util/concurrent",
        "//java/com/google/devtools/build/runtime:runfiles",
        "//java/com/google/geo/gt/util/flume:flume_table",
        "//java/com/google/io/file",
        "//java/com/google/io/file:googlefile_core",
        "//java/com/google/net/grpc:channel",
        "//java/com/google/pipeline/flume/fj:without_monitoring",
        "//java/com/google/pipeline/flume/util",
        "//java/com/google/protobuf",
        "//java/com/google/tech/file",
        "//java/com/google/thirdparty/grpc:javatime",
        "//third_party/java/auto:auto_value",
        "//third_party/java/checker_framework_annotations",
        "//third_party/java/error_prone:annotations",
        "//third_party/java/flogger",
        "//third_party/java/grpc:core",
    ],
)
