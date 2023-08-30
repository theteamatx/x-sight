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

import com.google.analysis.dremel.core.capacitor.CapacitorException;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.googlex.cortex.sight.Block;
import com.google.googlex.cortex.sight.Collections;
import com.google.googlex.cortex.sight.Sight;
import com.google.googlex.cortex.sight.SightImpl;
import com.google.protos.sight.x.proto.Sight.Params;
import java.io.IOException;

/**
 * Demonstration of representative Sight logging functionality that highlights the use of the
 * functionality to log data structures.
 */
final class CollectionsDemo {
  public static void main(String[] args) throws CapacitorException, IOException {
    try (SightImpl sight =
        SightImpl.create(
            Params.newBuilder()
                .setLocal(true)
                .setCapacitorOutput(true)
                .setLogOwner("bronevet")
                .setLogDirPath("/tmp/")
                .build())) {

      try (Block bList = Block.create("List", sight)) {
        Collections.log(ImmutableList.of("a", "b", "c", "d", "e"), sight);
      }

      try (Block bMap = Block.create("Map", sight)) {
        Collections.log(ImmutableMap.of(1, "a", 2, "b", 3, "c", 4, "d", 5, "e"), sight);
      }

      try (Block bDict = Block.create("Dict", sight)) {
        Collections.log(
            ImmutableMap.of(
                "key1",
                ImmutableList.of("value1a", "value1b"),
                "key2",
                ImmutableList.of("value2a", "value2b", "value2c")),
            sight);
      }

      try (Block bScalars = Block.create("Scalars", sight)) {
        Collections.log(12345, sight);
        Sight.text("///", sight);
        Collections.log(1.5, sight);
        Sight.text("///", sight);
        Collections.log("str", sight);
      }

      try (Block bNamedValue = Block.create("NamedValue", sight)) {
        Collections.log("var", 1.7, sight);
        Collections.log("var", 14.2, sight);
        Collections.log("x", ImmutableMap.of(1, "val1", 25, "val25"), sight);
      }

      try (Block bDeep = Block.create("Deep", sight)) {
        Collections.log(
            ImmutableList.of(
                ImmutableMap.of(
                    ImmutableList.of(1, 2),
                    1.5,
                    ImmutableList.of(12, 17, 23, 6),
                    14,
                    ImmutableList.of(7, 1, 1, 1),
                    2.5)),
            sight);
      }

      try (Block bTable = Block.create("Table", sight)) {
        Collections.logTable(
            /* keyColumns= */ ImmutableList.of("col1"),
            /* valueColumns= */ ImmutableList.of("col2", "col3"),
            /* rows= */ ImmutableList.of(
                new double[] {1, 2, 3}, new double[] {3, 4, 9}, new double[] {4, 8, 27}),
            sight);
      }
    }
  }

  private CollectionsDemo() {}
}
