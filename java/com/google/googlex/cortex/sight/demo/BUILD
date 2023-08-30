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

# Binaries that demonstrate various aspects of Sight's logging functionality.

ENVIRONMENT_JAVA_LAUNCHER = "//devtools/java/launcher:envelope_loader"

java_binary(
    name = "Demo",
    srcs = [
        "Demo.java",
    ],
    data = [
        "//googlex/cortex/sight/proto:sight_proto2db",
    ],
    launcher = ENVIRONMENT_JAVA_LAUNCHER,
    main_class = "com.google.googlex.cortex.sight.demo.Demo",
    # borgmaster will reject RPC's from unstamped binaries. Make sure this is always stamped even
    # if a --nostamp flag creeps in somewhere.
    stamp = 1,
    runtime_deps = [
        "//googlex/cortex/sight/proto:sight_proto2db",
        "//java/com/google/io/file:googlefile_cns",
        "//java/com/google/tech/file",
    ],
    deps = [
        "//googlex/cortex/sight/proto:sight_java_proto",
        "//java/com/google/analysis/dremel/core/capacitor:common",
        "//java/com/google/common/collect",
        "//java/com/google/googlex/cortex/sight:Sight",
    ],
)

java_binary(
    name = "CollectionsDemo",
    srcs = [
        "CollectionsDemo.java",
    ],
    data = [
        "//googlex/cortex/sight/proto:sight_proto2db",
    ],
    launcher = ENVIRONMENT_JAVA_LAUNCHER,
    main_class = "com.google.googlex.cortex.sight.demo.CollectionsDemo",
    # borgmaster will reject RPC's from unstamped binaries. Make sure this is always stamped even
    # if a --nostamp flag creeps in somewhere.
    stamp = 1,
    runtime_deps = [
        "//googlex/cortex/sight/proto:sight_proto2db",
        "//java/com/google/io/file:googlefile_cns",
        "//java/com/google/tech/file",
    ],
    deps = [
        "//googlex/cortex/sight/proto:sight_java_proto",
        "//java/com/google/analysis/dremel/core/capacitor:common",
        "//java/com/google/common/collect",
        "//java/com/google/googlex/cortex/sight:Sight",
    ],
)

java_binary(
    name = "FlumeDemo",
    srcs = [
        "FlumeDemo.java",
    ],
    data = [
        "//googlex/cortex/sight/proto:sight_proto2db",
    ],
    launcher = ENVIRONMENT_JAVA_LAUNCHER,
    main_class = "com.google.googlex.cortex.sight.demo.FlumeDemo",
    # borgmaster will reject RPC's from unstamped binaries. Make sure this is always stamped even
    # if a --nostamp flag creeps in somewhere.
    stamp = 1,
    runtime_deps = [
        "//googlex/cortex/sight/proto:sight_proto2db",
        "//java/com/google/io/file:googlefile_cns",
        "//java/com/google/tech/file",
    ],
    deps = [
        "//googlex/cortex/sight/proto:sight_java_proto",
        "//java/com/google/common/base",
        "//java/com/google/googlex/cortex/sight:Sight",
        "//java/com/google/pipeline/flume/fj",  # fixdeps: keep
        "//java/com/google/pipeline/flume/fj:without_monitoring",
    ],
)

java_binary(
    name = "FlumeJoinDemo",
    srcs = [
        "FlumeJoinDemo.java",
    ],
    data = [
        "//googlex/cortex/sight/proto:sight_proto2db",
    ],
    launcher = ENVIRONMENT_JAVA_LAUNCHER,
    main_class = "com.google.googlex.cortex.sight.demo.FlumeJoinDemo",
    # borgmaster will reject RPC's from unstamped binaries. Make sure this is always stamped even
    # if a --nostamp flag creeps in somewhere.
    stamp = 1,
    runtime_deps = [
        "//googlex/cortex/sight/proto:sight_proto2db",
        "//java/com/google/io/file:googlefile_cns",
        "//java/com/google/tech/file",
    ],
    deps = [
        "//googlex/cortex/sight/proto:sight_java_proto",
        "//java/com/google/common/base",
        "//java/com/google/googlex/cortex/sight:Sight",
        "//java/com/google/pipeline/flume/fj",  # fixdeps: keep
        "//java/com/google/pipeline/flume/fj:without_monitoring",
    ],
)
