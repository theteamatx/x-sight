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
package com.google.googlex.cortex.sight;

import static com.google.common.collect.ImmutableList.toImmutableList;
import static java.util.stream.Collectors.joining;

import java.util.ArrayDeque;
import java.util.Arrays;
import java.util.concurrent.atomic.AtomicInteger;

/** Unique IDs for locations within the Sight log */
public final class Location {
  private ArrayDeque<AtomicInteger> id;

  public static Location create() {
    Location loc = new Location();
    loc.id = new ArrayDeque<>();
    loc.id.push(new AtomicInteger(0));
    return loc;
  }

  public static Location create(String serializedLoc) {
    Location loc = new Location();
    loc.id = new ArrayDeque<>();
    loc.id.addAll(
        Arrays.stream(serializedLoc.split(":"))
            .map(Integer::parseInt)
            .map(AtomicInteger::new)
            .collect(toImmutableList()));
    return loc;
  }

  public static Location create(ArrayDeque<AtomicInteger> id) {
    Location loc = new Location();
    loc.id = id;
    return loc;
  }

  public void enter(int deeperId) {
    id.addLast(new AtomicInteger(deeperId));
  }

  public void exit() {
    id.removeLast();
  }

  public void next() {
    id.peekLast().incrementAndGet();
  }

  public void nextAll() {
    for (AtomicInteger pos : id) {
      pos.incrementAndGet();
    }
  }

  public int pos() {
    return id.peekLast().get();
  }

  public boolean isEmpty() {
    return id.isEmpty();
  }

  public int size() {
    return id.size();
  }

  @Override
  public String toString() {
    return id.stream().map(i -> String.format("%010d", i.get())).collect(joining(":"));
  }
}
