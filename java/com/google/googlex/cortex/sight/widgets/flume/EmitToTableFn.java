/*
 * Copyright 2023 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package com.google.googlex.cortex.sight.widgets.flume;

import com.google.common.base.Pair;
import com.google.errorprone.annotations.ResultIgnorabilityUnspecified;
import com.google.googlex.cortex.sight.Location;
import com.google.googlex.cortex.sight.Sight;
import com.google.pipeline.flume.fj.FJ;
import com.google.protos.sight.x.proto.Sight.Object.SubType;
import com.google.protos.sight.x.widgets.flume.proto.Flume.FlumeDoFnEmit;
import java.util.Optional;

/** Wraps Flume EmitToTableFn class with a Sight loggable variant. */
public class EmitToTableFn<K0, V0> extends FJ.EmitFn<Pair<K0, V0>> {
  private long stageId;
  private Optional<FJ.EmitToTableFn<K0, Pair<SightId, V0>>> baseEmit;
  private long itemId;
  private Sight sight;

  public static <K0, V0> EmitToTableFn<K0, V0> create(
      long stageId, FJ.EmitToTableFn<K0, Pair<SightId, V0>> baseEmit, Sight sight) {
    EmitToTableFn<K0, V0> emitFn = new EmitToTableFn<>();
    emitFn.stageId = stageId;
    emitFn.baseEmit = Optional.of(baseEmit);
    emitFn.itemId = 0;
    emitFn.sight = sight;
    return emitFn;
  }

  public static <K0, V0> EmitToTableFn<K0, V0> create(long stageId, Sight sight) {
    EmitToTableFn<K0, V0> emitFn = new EmitToTableFn<>();
    emitFn.stageId = stageId;
    emitFn.baseEmit = Optional.empty();
    emitFn.itemId = 0;
    emitFn.sight = sight;
    return emitFn;
  }

  @ResultIgnorabilityUnspecified
  public Location emitEmitFn() {
    return sight.logObject(
        /* advanceLocation= */ true,
        Optional.of(Thread.currentThread().getStackTrace()[3]),
        /* text= */ "",
        com.google.protos.sight.x.proto.Sight.Object.newBuilder()
            .setSubType(SubType.ST_FLUME_DO_FN_EMIT)
            .setFlumeDoFnEmit(FlumeDoFnEmit.newBuilder().setStageId(stageId).setItemId(itemId)));
  }

  public void advanceItemId() {
    ++itemId;
  }

  public Pair<K0, Pair<SightId, V0>> getLoggedOutput(K0 outKey, V0 outValue) {
    return Pair.of(outKey, Pair.of(SightId.create(stageId, itemId), outValue));
  }

  public void emit(K0 outKey, V0 outValue) {
    emitEmitFn();
    if (baseEmit.isPresent()) {
      baseEmit.get().emit(getLoggedOutput(outKey, outValue));
    }
    advanceItemId();
  }

  @Override
  public void emit(Pair<K0, V0> output) {
    emitEmitFn();
    if (baseEmit.isPresent()) {
      baseEmit.get().emit(getLoggedOutput(output.getFirst(), output.getSecond()));
    }
    advanceItemId();
  }
}
