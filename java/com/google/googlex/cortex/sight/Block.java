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

package com.google.googlex.cortex.sight;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import java.util.Map;
import org.checkerframework.checker.nullness.qual.Nullable;

/**
 * Class makes it easy to match the lifetimes of blocks in the log to lexical scopes in the source
 * code since they will match the lifetime of Block-type variable objects.
 */
public class Block implements AutoCloseable {
  private String label;

  private ImmutableList<String> attributeKeys;

  private @Nullable Sight sight;

  /** The {@code Location} of this {@code Block}'s starting point in the log. */
  private Location entryLocation;

  public static @Nullable Block create(String label, @Nullable Sight sight) {
    if (sight == null) {
      return null;
    }
    return create(
        label,
        /* attributes= */ ImmutableMap.of(),
        /* locationOfLogEvent= */ Sight.getCallerStackTraceElement(),
        sight);
  }

  public static @Nullable Block create(
      String label, Map<String, String> attributes, @Nullable Sight sight) {
    if (sight == null) {
      return null;
    }
    return create(
        label, attributes, /* locationOfLogEvent= */ Sight.getCallerStackTraceElement(), sight);
  }

  public static @Nullable Block create(String attrKey, String attrValue, @Nullable Sight sight) {
    if (sight == null) {
      return null;
    }
    return create(
        String.format("%s = %s", attrKey, attrValue),
        /* attributes= */ ImmutableMap.of(attrKey, attrValue),
        /* locationOfLogEvent= */ Sight.getCallerStackTraceElement(),
        sight);
  }

  public static @Nullable Block create(String attrKey, int attrValue, @Nullable Sight sight) {
    return create(attrKey, (long) attrValue, sight);
  }

  public static @Nullable Block create(String attrKey, long attrValue, @Nullable Sight sight) {
    if (sight == null) {
      return null;
    }
    return create(
        String.format("%s = %d", attrKey, attrValue),
        /* attributes= */ ImmutableMap.of(attrKey, Long.toString(attrValue)),
        /* locationOfLogEvent= */ Sight.getCallerStackTraceElement(),
        sight);
  }

  public static Block create(String label, StackTraceElement locationOfLogEvent, Sight sight) {
    return create(label, ImmutableMap.of(), locationOfLogEvent, sight);
  }

  protected static Block create(
      String label,
      Map<String, String> attributes,
      StackTraceElement locationOfLogEvent,
      Sight sight) {
    Block b = new Block();
    initialize(
        b,
        label,
        attributes,
        locationOfLogEvent,
        com.google.protos.sight.x.proto.Sight.Object.newBuilder(),
        sight);
    return b;
  }

  protected static void initialize(
      Block b,
      String label,
      Map<String, String> attributes,
      StackTraceElement locationOfLogEvent,
      com.google.protos.sight.x.proto.Sight.Object.Builder object,
      Sight sight) {
    b.label = label;
    b.attributeKeys = ImmutableList.copyOf(attributes.keySet());
    b.sight = sight;
    for (String key : b.attributeKeys) {
      sight.setAttribute(key, attributes.get(key));
    }
    b.entryLocation = sight.enterBlock(label, locationOfLogEvent, object);
  }

  /** Returns the {@code Location} of this {@code Block}'s starting point in the log. */
  public Location getEntryLocation() {
    return entryLocation;
  }

  protected void closeWithObject(com.google.protos.sight.x.proto.Sight.Object.Builder object) {
    // Note: we use the 3rd stack frame since closeWithObject() is called from functions
    // (e.g. close() below), which are called from user code.
    sight.exitBlock(label, Thread.currentThread().getStackTrace()[3], object);
    for (String key : attributeKeys.reverse()) {
      sight.unsetAttribute(key);
    }
  }

  @Override
  public void close() {
    closeWithObject(com.google.protos.sight.x.proto.Sight.Object.newBuilder());
  }
}
