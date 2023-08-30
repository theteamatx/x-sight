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

import static com.google.common.collect.ImmutableList.toImmutableList;

import com.google.common.base.Pair;
import com.google.pipeline.flume.fj.FJ;
import com.google.pipeline.flume.fj.FJ.MapFn;
import com.google.pipeline.flume.fj.PDataType;
import java.util.Collection;

/** Utilities for managing Flume Java PCollections that include SightIds along with the data. */
public final class PCollection {
  /**
   * Returns a {@code PCollection} of logged data items (include a zero {@code SightId}) with the
   * same data payload as in sequential {@code Collection} {@code elems}.
   */
  public static <T> com.google.pipeline.flume.fj.PCollection<Pair<SightId, T>> create(
      Collection<T> elems, PDataType<T> pd) {
    return com.google.pipeline.flume.fj.PCollection.create(
        elems.stream().map(e -> Pair.of(SightId.zero(), e)).collect(toImmutableList()),
        FJ.collectionOf(FJ.pairsOf(SightId.getPDataType(), pd)));
  }

  /**
   * Returns a {@code PCollection} of logged data items (include a zero {@code SightId}) with the
   * same data payload as in {@code elems}.
   */
  public static <T> com.google.pipeline.flume.fj.PCollection<Pair<SightId, T>> logged(
      com.google.pipeline.flume.fj.PCollection<T> elems) {
    return elems.parallelDo(
        new FJ.KeyFn<SightId, T>() {

          @Override
          public SightId key(T elem) {
            return SightId.zero();
          }
        });
  }

  /**
   * Returns a {@code PCollection} of raw (unlogged) logged data items with the same data payload as
   * the logged items (include a {@code SightId}) as in {@code elems}.
   */
  public static <T> com.google.pipeline.flume.fj.PCollection<T> unlogged(
      com.google.pipeline.flume.fj.PCollection<Pair<SightId, T>> elems) {
    return elems.parallelDo(
        new MapFn<Pair<SightId, T>, T>() {
          @Override
          public T map(Pair<SightId, T> elem) {
            return elem.getSecond();
          }
        });
  }

  private PCollection() {}
}
