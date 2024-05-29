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
package com.google.googlex.cortex.sight.demo;

import com.google.analysis.dremel.core.capacitor.CapacitorException;
import com.google.common.collect.ImmutableMap;
import com.google.googlex.cortex.sight.Attribute;
import com.google.googlex.cortex.sight.Block;
import com.google.googlex.cortex.sight.Sight;
import com.google.googlex.cortex.sight.SightImpl;
import com.google.protos.sight.x.proto.Sight.Params;
import java.io.IOException;

/**
 * Demonstration of representative Sight logging functionality that highlights the use of the major
 * types of log elements.
 */
final class Demo {
  public static void main(String[] args) throws CapacitorException, IOException {
    try (SightImpl sight =
        SightImpl.create(
            Params.newBuilder()
                // .setLocal(true)
                .setCapacitorOutput(true)
                .setLogOwner("bronevet")
                // .setLogDirPath("/tmp/")
                .build())) {

      try (Block ba = Block.create("A", ImmutableMap.of("key", "A"), sight)) {
        Sight.textLine("A preText", sight);
        try (Block ba1 = Block.create("A1", sight)) {
          try (Block ba11 = Block.create("A1.1", sight)) {
            Sight.textLine("A1.1 text", sight);
          }
          Sight.textLine("A text", sight);
        }
        Sight.textLine("A postText", sight);
      }
      try (Block bb = Block.create("B", sight)) {
        Sight.textLine("B preText", sight);
        try (Attribute ab = Attribute.create(ImmutableMap.of("key", "B", "key1", "B"), sight)) {
          try (Attribute ab3 = Attribute.create("key2", "B", sight)) {
            Sight.textLine("B1 preText", sight);
            try (Block bb1 = Block.create("B1", sight)) {
              try (Block bb11 = Block.create("B1.1", sight)) {
                Sight.textLine("B1.1 text", sight);
              }
              Sight.textLine("B1 text", sight);
            }
            Sight.textLine("B1 postText", sight);
          }
          try (Block b2 =
              Block.create(
                  "B2",
                  ImmutableMap.of("keyin", "X", "keyin1", "X", "keyin2", "X", "keyin3", "X"),
                  sight)) {
            try (Block b21 = Block.create("B2.1", sight)) {
              Sight.textLine("B2.1 text", sight);
            }
            Sight.textLine("B2 text", sight);
          }
          try (Block b3 = Block.create("B3", sight)) {
            try (Block b31 = Block.create("B3.1", sight)) {
              Sight.textLine("B3.1 text", sight);
            }
            Sight.textLine("B3 text", sight);
          }
        }
        Sight.textLine("B postText", sight);
      }
    }
  }

  private Demo() {}
}
