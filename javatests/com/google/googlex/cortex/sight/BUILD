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

# Tests Sight logging infrastructure.

java_test(
    name = "SightTest",
    srcs = ["SightTest.java"],
    data = [
        "//googlex/cortex/sight/proto:sight_proto2db",
    ],
    test_class = "com.google.googlex.cortex.sight.SightTest",
    deps = [
        "//googlex/cortex/sight/proto:sight_java_proto",
        "//googlex/cortex/sight/service/v1:sight_service_java_grpc",
        "//googlex/cortex/sight/service/v1:sight_service_java_proto",
        "//java/com/google/analysis/dremel/core/capacitor:common",
        "//java/com/google/analysis/dremel/core/capacitor:record-reader",
        "//java/com/google/common/collect",
        "//java/com/google/common/time/testing",
        "//java/com/google/googlex/cortex/sight:Sight",
        "//java/com/google/protobuf",
        "//java/com/google/testing/util:asserting_handler",
        "//third_party/java/apache_commons_io",
        "//third_party/java/flogger",
        "//third_party/java/grpc:context",
        "//third_party/java/grpc:stub",
        "//third_party/java/grpc:testing",
        "//third_party/java/junit",
        "//third_party/java/mockito",
        "//third_party/java/truth",
        "//third_party/java/truth/extensions:proto",
    ],
)

java_test(
    name = "SightCollectionsTest",
    srcs = ["SightCollectionsTest.java"],
    data = [
        "//googlex/cortex/sight/proto:sight_proto2db",
    ],
    test_class = "com.google.googlex.cortex.sight.SightCollectionsTest",
    deps = [
        "//googlex/cortex/sight/proto:sight_java_proto",
        "//java/com/google/analysis/dremel/core/capacitor:common",
        "//java/com/google/analysis/dremel/core/capacitor:record-reader",
        "//java/com/google/common/collect",
        "//java/com/google/common/time/testing",
        "//java/com/google/googlex/cortex/sight:Sight",
        "//java/com/google/protobuf",
        "//third_party/java/apache_commons_io",
        "//third_party/java/flogger",
        "//third_party/java/junit",
        "//third_party/java/mockito",
        "//third_party/java/truth",
        "//third_party/java/truth/extensions:proto",
    ],
)

java_test(
    name = "SightProtoTest",
    srcs = ["SightProtoTest.java"],
    data = [
        "//googlex/cortex/sight/proto:sight_proto2db",
    ],
    test_class = "com.google.googlex.cortex.sight.SightProtoTest",
    deps = [
        ":sight_test_mesg_java_proto",
        "//googlex/cortex/sight/proto:sight_java_proto",
        "//java/com/google/analysis/dremel/core/capacitor:common",
        "//java/com/google/analysis/dremel/core/capacitor:record-reader",
        "//java/com/google/common/collect",
        "//java/com/google/common/time/testing",
        "//java/com/google/googlex/cortex/sight:Sight",
        "//java/com/google/protobuf",
        "//third_party/java/apache_commons_io",
        "//third_party/java/flogger",
        "//third_party/java/junit",
        "//third_party/java/mockito",
        "//third_party/java/truth",
        "//third_party/java/truth/extensions:proto",
    ],
)

# Proto libraries
proto_library(
    name = "sight_test_mesg_proto",
    srcs = [
        "sight_test_mesg.proto",
    ],
    cc_api_version = 2,
    alwayslink = 1,
)

java_proto_library(
    name = "sight_test_mesg_java_proto",
    deps = [":sight_test_mesg_proto"],
)
