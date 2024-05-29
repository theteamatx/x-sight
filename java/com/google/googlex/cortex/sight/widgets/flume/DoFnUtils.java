// Copyright 2023 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
package com.google.googlex.cortex.sight.widgets.flume;

import com.google.googlex.cortex.sight.Location;
import com.google.googlex.cortex.sight.Sight;
import com.google.protos.sight.x.proto.Sight.BlockEnd;
import com.google.protos.sight.x.proto.Sight.BlockStart;
import com.google.protos.sight.x.widgets.flume.proto.Flume.FlumeDoFnComplete;
import com.google.protos.sight.x.widgets.flume.proto.Flume.FlumeDoFnCreate;

/** Functionality common to Sight DoFn overrides. */
final class DoFnUtils {
  public static Location emitDoFnCreate(String label, Sight sight) {
    return sight.enterBlock(
        label,
        Thread.currentThread().getStackTrace()[3],
        com.google.protos.sight.x.proto.Sight.Object.newBuilder()
            .setBlockStart(
                BlockStart.newBuilder()
                    .setSubType(BlockStart.SubType.ST_FLUME_DO_FN_CREATE)
                    .setFlumeDoFnCreate(FlumeDoFnCreate.newBuilder().setLabel(label))));
  }

  public static void emitDoFnComplete(String label, Sight sight) {
    sight.exitBlock(
        label,
        Thread.currentThread().getStackTrace()[3],
        com.google.protos.sight.x.proto.Sight.Object.newBuilder()
            .setBlockEnd(
                BlockEnd.newBuilder()
                    .setSubType(BlockEnd.SubType.ST_FLUME_DO_FN_COMPLETE)
                    .setFlumeDoFnComplete(FlumeDoFnComplete.newBuilder().setLabel(label))));
  }

  private DoFnUtils() {}
}
