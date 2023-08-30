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

package com.google.googlex.cortex.sight.demo;

import com.google.common.base.Pair;
import com.google.common.collect.ImmutableList;
import com.google.googlex.cortex.sight.Sight;
import com.google.googlex.cortex.sight.SightImpl;
import com.google.googlex.cortex.sight.widgets.flume.DoToTableFn;
import com.google.googlex.cortex.sight.widgets.flume.EmitToTableFn;
import com.google.googlex.cortex.sight.widgets.flume.FlumeUtils;
import com.google.googlex.cortex.sight.widgets.flume.PCollection;
import com.google.googlex.cortex.sight.widgets.flume.SightId;
import com.google.pipeline.flume.fj.FJ;
import com.google.pipeline.flume.fj.FlumeJava;
import com.google.pipeline.flume.util.Tuple2;
import com.google.protos.sight.x.proto.Sight.Params;
import java.util.List;

/** Demonstrates logging Flume pipelines with joins. */
final class FlumeJoinDemo {
  public static class ToNumStringTable extends DoToTableFn<Integer, Integer, String> {
    public ToNumStringTable(Sight sight) {
      super(sight, /* derivedClassName= */ "ToNumStringTable");
    }

    @Override
    public void process(Integer input, EmitToTableFn<Integer, String> emitFn, Sight sight) {
      sight.textLine(String.format("input=%s", input));
      emitFn.emit(input, String.format("%d", input));
    }
  }

  public static class ToNumBracketedStringTable extends DoToTableFn<Integer, Integer, String> {
    public ToNumBracketedStringTable(Sight sight) {
      super(sight, /* derivedClassName= */ "ToNumBracketedStringTable");
    }

    @Override
    public void process(Integer input, EmitToTableFn<Integer, String> emitFn, Sight sight) {
      sight.textLine(String.format("input=%s", input));
      emitFn.emit(input, String.format("[%d]", input));
    }
  }

  public static class Merge
      extends DoToTableFn<Pair<Integer, Tuple2.Collections<String, String>>, Integer, String> {
    public Merge(Sight sight) {
      super(sight, /* derivedClassName= */ "Merge");
    }

    @Override
    public void process(
        Pair<Integer, Tuple2.Collections<String, String>> input,
        EmitToTableFn<Integer, String> emitFn,
        Sight sight) {
      sight.textLine(String.format("v0=%s", input.getSecond().elements().get(0)));
      sight.textLine(String.format("v1=%s", input.getSecond().elements().get(1)));
      @SuppressWarnings("unchecked") // Type guaranteed by Flume's workflow composition rules.
      List<String> v0 = (List<String>) input.getSecond().elements().get(0);
      @SuppressWarnings("unchecked") // Type guaranteed by Flume's workflow composition rules.
      List<String> v1 = (List<String>) input.getSecond().elements().get(1);
      sight.textLine(String.format("%d: %s / %s", input.getFirst(), v0, v1));
      emitFn.emit(input.getFirst(), String.format("%d: %s / %s", input.getFirst(), v0, v1));
    }
  }

  public static void main(String[] args) throws Exception {
    FlumeJava.init(args);

    try (SightImpl sight =
        SightImpl.create(
            Params.newBuilder()
                .setLocal(true)
                .setCapacitorOutput(true)
                .setLogOwner("bronevet")
                .setLogDirPath("/tmp/")
                .build())) {

      com.google.pipeline.flume.fj.PCollection<Pair<SightId, Integer>> numbers =
          PCollection.create(
              ImmutableList.of(
                  Integer.valueOf(1),
                  Integer.valueOf(2),
                  Integer.valueOf(3),
                  Integer.valueOf(4),
                  Integer.valueOf(5),
                  Integer.valueOf(6),
                  Integer.valueOf(7),
                  Integer.valueOf(8),
                  Integer.valueOf(9),
                  Integer.valueOf(10),
                  Integer.valueOf(11),
                  Integer.valueOf(12)),
              FJ.ints());
      FlumeJava.run(
          FlumeUtils.innerJoin(
                  FlumeUtils.innerJoin(
                          numbers.parallelDo(new ToNumStringTable(sight)),
                          numbers.parallelDo(new ToNumBracketedStringTable(sight)),
                          sight)
                      .parallelDo(new Merge(sight)),
                  FlumeUtils.innerJoin(
                          numbers.parallelDo(new ToNumStringTable(sight)),
                          numbers.parallelDo(new ToNumBracketedStringTable(sight)),
                          sight)
                      .parallelDo(new Merge(sight)),
                  sight)
              .parallelDo(new Merge(sight)));
      FlumeJava.done();
    }
  }

  private FlumeJoinDemo() {}
}
