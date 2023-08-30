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
import com.google.pipeline.flume.fj.FJ.DoToTableFn;
import com.google.pipeline.flume.fj.FJ.EmitToTableFn;

/** Utilities for managing Flume Java PCollections that include SightIds along with the data. */
public final class PTable {
  /**
   * Returns a {@code PTable} of logged data items (include a zero {@code SightId}) with the same
   * data payload as in {@code elems}.
   */
  public static <K, V> com.google.pipeline.flume.fj.PTable<K, Pair<SightId, V>> logged(
      com.google.pipeline.flume.fj.PTable<K, V> elems) {
    return elems.parallelDo(
        new DoToTableFn<Pair<K, V>, K, Pair<SightId, V>>() {
          @Override
          public void process(Pair<K, V> elem, EmitToTableFn<K, Pair<SightId, V>> emitFn) {
            emitFn.emit(elem.getFirst(), Pair.of(SightId.zero(), elem.getSecond()));
          }
        });
  }

  /**
   * Returns a {@code PCollection} of raw (unlogged) logged data items with the same data payload as
   * the logged items (include a {@code SightId}) as in {@code elems}.
   */
  public static <K, V> com.google.pipeline.flume.fj.PTable<K, V> unlogged(
      com.google.pipeline.flume.fj.PTable<K, Pair<SightId, V>> elems) {
    return elems.parallelDo(
        new DoToTableFn<Pair<K, Pair<SightId, V>>, K, V>() {
          @Override
          public void process(Pair<K, Pair<SightId, V>> elem, EmitToTableFn<K, V> emitFn) {
            emitFn.emit(elem.getFirst(), elem.getSecond().getSecond());
          }
        });
  }

  private PTable() {}
}
