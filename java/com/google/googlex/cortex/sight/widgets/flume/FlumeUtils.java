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

import static com.google.common.collect.ImmutableList.toImmutableList;

import com.google.analysis.dremel.core.capacitor.CapacitorException;
import com.google.common.base.Pair;
import com.google.common.flogger.GoogleLogger;
import com.google.geo.gt.util.flume.FlumeTable;
import com.google.googlex.cortex.sight.Location;
import com.google.googlex.cortex.sight.Sight;
import com.google.googlex.cortex.sight.SightImpl;
import com.google.pipeline.flume.fj.FJ;
import com.google.pipeline.flume.util.Tuple2;
import com.google.protos.sight.x.proto.Sight.Params;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import org.checkerframework.checker.nullness.qual.Nullable;

/** Wrapper for functionality in com.google.geo.gt.util.FlumeUtils. */
public final class FlumeUtils {
  private static final GoogleLogger logger = GoogleLogger.forEnclosingClass();

  private static class JoinMergerFn<K, V0, V1>
      extends FJ.DoToTableFn<
          Pair<K, Tuple2.Collections<Pair<SightId, V0>, Pair<SightId, V1>>>,
          K,
          Pair<SightId, Tuple2.Collections<V0, V1>>> {
    private final Params params;

    private @Nullable SightImpl sight;

    private @Nullable EmitToTableFn<K, Tuple2.Collections<V0, V1>> emitFn;

    public JoinMergerFn(Sight baseSight) {
      synchronized (baseSight) {
        Location location = DoFnUtils.emitDoFnCreate("Sight.FlumeUtils.JoinMergerFn", baseSight);
        params =
            baseSight.getParams().toBuilder()
                .setLocal(true)
                .setContainerLocation(location.toString())
                .build();
        sight = null;
        emitFn = null;

        DoFnUtils.emitDoFnComplete("Sight.FlumeUtils.JoinMergerFn", baseSight);
      }
    }

    public JoinMergerFn(Params params) {
      this.params = params;
    }

    @Override
    public void initialize() {
      try {
        sight = SightImpl.create(params);
      } catch (CapacitorException | IOException e) {
        logger.atSevere().withCause(e).log("While creating SightImpl");
      }
      sight.getLocation().enter(sight.hashCode());
    }

    @Override
    public void terminate() {
      sight.getLocation().exit();
      sight.close();
    }

    @Override
    public void process(
        Pair<K, Tuple2.Collections<Pair<SightId, V0>, Pair<SightId, V1>>> input,
        FJ.EmitToTableFn<K, Pair<SightId, Tuple2.Collections<V0, V1>>> baseEmitFn) {
      if (emitFn == null) {
        emitFn =
            EmitToTableFn.create(/* stageId= */ System.identityHashCode(this), baseEmitFn, sight);
      }

      @SuppressWarnings("unchecked") // Type guaranteed by Flume's workflow composition rules.
      ArrayList<Pair<SightId, V0>> v0 =
          (ArrayList<Pair<SightId, V0>>) input.getSecond().elements().get(0);
      @SuppressWarnings("unchecked") // Type guaranteed by Flume's workflow composition rules.
      ArrayList<Pair<SightId, V1>> v1 =
          (ArrayList<Pair<SightId, V1>>) input.getSecond().elements().get(1);
      ArrayList<DoFnProcessBlock> blocks = new ArrayList<>();
      blocks.addAll(
          v0.stream()
              .map(
                  i ->
                      DoFnProcessBlock.create(
                          "JoinMergerFn", /* isPassthrough= */ true, i.getFirst(), sight))
              .collect(toImmutableList()));
      blocks.addAll(
          v1.stream()
              .map(
                  i ->
                      DoFnProcessBlock.create(
                          "JoinMergerFn", /* isPassthrough= */ true, i.getFirst(), sight))
              .collect(toImmutableList()));
      emitFn.emit(
          input.getFirst(),
          Tuple2.Collections.of(
              v0.stream().map(Pair::getSecond).collect(toImmutableList()),
              v1.stream().map(Pair::getSecond).collect(toImmutableList())));
      Collections.reverse(blocks);
      for (DoFnProcessBlock b : blocks) {
        b.close();
      }
    }
  }

  public static <K, V0, V1>
      com.google.pipeline.flume.fj.PCollection<Pair<SightId, Pair<K, Tuple2.Collections<V0, V1>>>>
          innerJoin(
              com.google.pipeline.flume.fj.PTable<K, Pair<SightId, V0>> t0,
              com.google.pipeline.flume.fj.PTable<K, Pair<SightId, V1>> t1,
              Sight baseSight) {
    return FlumeTable.innerJoin(t0, t1)
        .parallelDo(new JoinMergerFn<K, V0, V1>(baseSight))
        .parallelDo(new TableToPairCollection<K, Tuple2.Collections<V0, V1>>());
  }

  /** Converts a PTable to a PCollection of key/value pairs. */
  private static class TableToPairCollection<K, V>
      extends FJ.MapFn<Pair<K, Pair<SightId, V>>, Pair<SightId, Pair<K, V>>> {
    @Override
    public Pair<SightId, Pair<K, V>> map(Pair<K, Pair<SightId, V>> input) {
      return Pair.of(
          input.getSecond().getFirst(), Pair.of(input.getFirst(), input.getSecond().getSecond()));
    }
  }

  private FlumeUtils() {}
}
