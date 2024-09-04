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

import com.google.common.collect.ImmutableMap;
import com.google.googlex.cortex.sight.Block;
import com.google.googlex.cortex.sight.Sight;
import com.google.protos.sight.x.proto.Sight.BlockEnd;
import com.google.protos.sight.x.proto.Sight.BlockStart;
import com.google.protos.sight.x.widgets.flume.proto.Flume.FlumeDoFnEndDo;
import com.google.protos.sight.x.widgets.flume.proto.Flume.FlumeDoFnStartDo;
import org.checkerframework.checker.nullness.qual.Nullable;

/** Variant of Sight Blocks that marks the scope of DoFn process() methods and their variants. */
final class DoFnProcessBlock extends Block implements AutoCloseable {
  private SightId sightId;

  public static @Nullable DoFnProcessBlock create(
      String label, SightId sightId, @Nullable Sight sight) {
    return create(label, /* isPassthrough= */ false, sightId, sight);
  }

  public static @Nullable DoFnProcessBlock create(
      String label, boolean isPassthrough, SightId sightId, @Nullable Sight sight) {
    if (sight == null) {
      return null;
    }
    DoFnProcessBlock b = new DoFnProcessBlock();
    initialize(
        b,
        label,
        /* attributes= */ ImmutableMap.of(),
        /* locationOfLogEvent= */ Sight.getCallerStackTraceElement(),
        com.google.protos.sight.x.proto.Sight.Object.newBuilder()
            .setBlockStart(
                BlockStart.newBuilder()
                    .setSubType(BlockStart.SubType.ST_FLUME_DO_FN_START_DO)
                    .setFlumeDoFnStartDo(
                        FlumeDoFnStartDo.newBuilder()
                            .setInputStageId(sightId.inputStageId())
                            .setInputItemId(sightId.inputItemId())
                            .setIsPassthrough(isPassthrough))),
        sight);
    b.sightId = sightId;
    return b;
  }

  @Override
  public void close() {
    closeWithObject(
        com.google.protos.sight.x.proto.Sight.Object.newBuilder()
            .setBlockEnd(
                BlockEnd.newBuilder()
                    .setSubType(BlockEnd.SubType.ST_FLUME_DO_FN_END_DO)
                    .setFlumeDoFnEndDo(
                        FlumeDoFnEndDo.newBuilder().setInputStageId(sightId.inputStageId()))));
  }

  private DoFnProcessBlock() {}
}
