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
import com.google.googlex.cortex.sight.widgets.flume.DoFn;
import com.google.googlex.cortex.sight.widgets.flume.EmitFn;
import com.google.googlex.cortex.sight.widgets.flume.PCollection;
import com.google.googlex.cortex.sight.widgets.flume.SightId;
import com.google.pipeline.flume.fj.FJ;
import com.google.pipeline.flume.fj.FlumeJava;
import com.google.protos.sight.x.proto.Sight.Params;

/** Demonstrates logging Flume pipelines. */
final class FlumeDemo {
  public static class DuplicateStr extends DoFn<String, String> {
    public DuplicateStr(Sight sight) {
      super(sight, /* derivedClassName= */ "DuplicateStr");
    }

    @Override
    public void process(String input, EmitFn<String> emitFn, Sight sight) {
      sight.textLine(String.format("input=%s", input));
      emitFn.emit(input + input);
    }
  }

  public static class Append extends DoFn<String, String> {
    public Append(Sight sight) {
      super(sight, /* derivedClassName= */ "Append");
    }

    @Override
    public void process(String input, EmitFn<String> emitFn, Sight sight) {
      sight.textLine(String.format("input=%s", input));
      emitFn.emit(input + ">");
    }
  }

  public static class Prepend extends DoFn<String, String> {
    public Prepend(Sight sight) {
      super(sight, /* derivedClassName= */ "Prepend");
    }

    @Override
    public void process(String input, EmitFn<String> emitFn, Sight sight) {
      sight.textLine(String.format("input=%s", input));
      emitFn.emit("<" + input);
    }
  }

  public static void main(String[] args) throws Exception {
    FlumeJava.init(args);

    try (SightImpl sight =
        SightImpl.create(
            Params.newBuilder()
                // .setLocal(true)
                .setCapacitorOutput(true)
                .setLogOwner("username@domain.com")
                // .setLogDirPath("/tmp/")
                .build())) {
      com.google.pipeline.flume.fj.PCollection<Pair<SightId, String>> numbers =
          PCollection.create(
              ImmutableList.of("1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12"),
              FJ.strings());
      FlumeJava.run(
          numbers
              .parallelDo(new DuplicateStr(sight))
              .parallelDo(new Append(sight))
              .parallelDo(new Prepend(sight)));
      FlumeJava.done();
    }
  }

  private FlumeDemo() {}
}
