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

import com.google.common.base.Pair;
import com.google.common.primitives.Doubles;
import com.google.errorprone.annotations.ResultIgnorabilityUnspecified;
import com.google.googlex.cortex.sight.widgets.protobuf.Proto;
import com.google.protobuf.MessageOrBuilder;
import com.google.protos.sight.x.proto.Sight.BlockEnd;
import com.google.protos.sight.x.proto.Sight.BlockStart;
import com.google.protos.sight.x.proto.Sight.ListStart;
import com.google.protos.sight.x.proto.Sight.Object.SubType;
import com.google.protos.sight.x.proto.Sight.Value;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import org.checkerframework.checker.nullness.qual.Nullable;

/** Class that makes it possible to log Java typed scalars and containers via Sight. */
public final class Collections {

  @ResultIgnorabilityUnspecified
  public static Location log(Object value, @Nullable Sight sight) {
    return log(value, Sight.getCallerStackTraceElement(), sight);
  }

  @ResultIgnorabilityUnspecified
  public static Location log(
      Object value, StackTraceElement locationOfLogEvent, @Nullable Sight sight) {
    if (sight == null) {
      return Location.create();
    }
    if (value instanceof SightLoggable) {
      return ((SightLoggable) value).log(locationOfLogEvent, sight);
    }
    if (value instanceof String) {
      return log((String) value, locationOfLogEvent, sight);
    }
    if (value instanceof Integer) {
      return log((Integer) value, locationOfLogEvent, sight);
    }
    if (value instanceof Long) {
      return log((Long) value, locationOfLogEvent, sight);
    }
    if (value instanceof Float) {
      return log((Float) value, locationOfLogEvent, sight);
    }
    if (value instanceof Double) {
      return log((Double) value, locationOfLogEvent, sight);
    }
    if (value instanceof double[]) {
      return logList(Doubles.asList((double[]) value), locationOfLogEvent, sight);
    }
    if (value instanceof List) {
      return logList((List<?>) value, locationOfLogEvent, sight);
    }
    if (value instanceof Pair) {
      return logPair((Pair<?, ?>) value, locationOfLogEvent, sight);
    }
    if (value instanceof Map) {
      // If the map has String keys, it best documented as a dictionary (map from labels to their
      // values).
      if (!((Map<?, ?>) value).isEmpty()
          && ((Map<?, ?>) value).entrySet().iterator().next().getKey() instanceof String) {
        @SuppressWarnings("unchecked") // Checked by conditional
        Map<String, ?> map = (Map<String, ?>) value;
        return logDict(map, locationOfLogEvent, sight);
      }
      return logMap((Map<?, ?>) value, locationOfLogEvent, sight);
    }
    if (value instanceof MessageOrBuilder) {
      Proto.log((MessageOrBuilder) value, sight);
    }
    return sight.textLine(value.toString(), locationOfLogEvent);
  }

  @ResultIgnorabilityUnspecified
  public static Location log(String name, Object value, @Nullable Sight sight) {
    return log(name, value, Sight.getCallerStackTraceElement(), sight);
  }

  @ResultIgnorabilityUnspecified
  public static Location log(
      String name, Object value, StackTraceElement locationOfLogEvent, @Nullable Sight sight) {
    if (sight == null) {
      return Location.create();
    }
    Location enterLoc =
        sight.enterBlock(
            name,
            locationOfLogEvent,
            com.google.protos.sight.x.proto.Sight.Object.newBuilder()
                .setBlockStart(
                    BlockStart.newBuilder().setSubType(BlockStart.SubType.ST_NAMED_VALUE)));
    log(value, locationOfLogEvent, sight);
    sight.exitBlock(
        name,
        locationOfLogEvent,
        com.google.protos.sight.x.proto.Sight.Object.newBuilder()
            .setBlockEnd(BlockEnd.newBuilder().setSubType(BlockEnd.SubType.ST_NAMED_VALUE)));
    return enterLoc;
  }

  @ResultIgnorabilityUnspecified
  public static <T> Location log(String value, @Nullable Sight sight) {
    return log(value, Sight.getCallerStackTraceElement(), sight);
  }

  public static <T> Location log(
      String value, StackTraceElement locationOfLogEvent, @Nullable Sight sight) {
    if (sight == null) {
      return Location.create();
    }
    return sight.logObject(
        /* advanceLocation= */ true,
        Optional.of(locationOfLogEvent),
        value + "\n",
        com.google.protos.sight.x.proto.Sight.Object.newBuilder()
            .setSubType(SubType.ST_VALUE)
            .setValue(
                Value.newBuilder().setSubType(Value.SubType.ST_STRING).setStringValue(value)));
  }

  @ResultIgnorabilityUnspecified
  public static <T> Location log(Integer value, @Nullable Sight sight) {
    return log(value, Sight.getCallerStackTraceElement(), sight);
  }

  public static <T> Location log(
      Integer value, StackTraceElement locationOfLogEvent, @Nullable Sight sight) {
    if (sight == null) {
      return Location.create();
    }
    return sight.logObject(
        /* advanceLocation= */ true,
        Optional.of(locationOfLogEvent),
        String.format("%d\n", value),
        com.google.protos.sight.x.proto.Sight.Object.newBuilder()
            .setSubType(SubType.ST_VALUE)
            .setValue(Value.newBuilder().setSubType(Value.SubType.ST_INT64).setInt64Value(value)));
  }

  public static <T> Location log(Long value, @Nullable Sight sight) {
    return log(value, Sight.getCallerStackTraceElement(), sight);
  }

  public static <T> Location log(
      Long value, StackTraceElement locationOfLogEvent, @Nullable Sight sight) {
    if (sight == null) {
      return Location.create();
    }
    return sight.logObject(
        /* advanceLocation= */ true,
        Optional.of(locationOfLogEvent),
        String.format("%d\n", value),
        com.google.protos.sight.x.proto.Sight.Object.newBuilder()
            .setSubType(SubType.ST_VALUE)
            .setValue(Value.newBuilder().setSubType(Value.SubType.ST_INT64).setInt64Value(value)));
  }

  public static <T> Location log(Float value, @Nullable Sight sight) {
    return log(value, Sight.getCallerStackTraceElement(), sight);
  }

  public static <T> Location log(
      Float value, StackTraceElement locationOfLogEvent, @Nullable Sight sight) {
    if (sight == null) {
      return Location.create();
    }
    return sight.logObject(
        /* advanceLocation= */ true,
        Optional.of(locationOfLogEvent),
        String.format("%f\n", value),
        com.google.protos.sight.x.proto.Sight.Object.newBuilder()
            .setSubType(SubType.ST_VALUE)
            .setValue(
                Value.newBuilder().setSubType(Value.SubType.ST_DOUBLE).setDoubleValue(value)));
  }

  @ResultIgnorabilityUnspecified
  public static <T> Location log(Double value, @Nullable Sight sight) {
    return log(value, Sight.getCallerStackTraceElement(), sight);
  }

  public static <T> Location log(
      Double value, StackTraceElement locationOfLogEvent, @Nullable Sight sight) {
    if (sight == null) {
      return Location.create();
    }
    return sight.logObject(
        /* advanceLocation= */ true,
        Optional.of(locationOfLogEvent),
        String.format("%f\n", value),
        com.google.protos.sight.x.proto.Sight.Object.newBuilder()
            .setSubType(SubType.ST_VALUE)
            .setValue(
                Value.newBuilder().setSubType(Value.SubType.ST_DOUBLE).setDoubleValue(value)));
  }

  private static <T> Location logList(
      List<T> list, StackTraceElement locationOfLogEvent, @Nullable Sight sight) {
    Location enterLoc =
        sight.enterBlock(
            "list",
            locationOfLogEvent,
            com.google.protos.sight.x.proto.Sight.Object.newBuilder()
                .setBlockStart(
                    BlockStart.newBuilder()
                        .setSubType(BlockStart.SubType.ST_LIST)
                        .setList(
                            ListStart.newBuilder().setSubType(ListStart.SubType.ST_HOMOGENEOUS))));
    list.forEach(e -> log(e, locationOfLogEvent, sight));
    sight.exitBlock(
        "list",
        locationOfLogEvent,
        com.google.protos.sight.x.proto.Sight.Object.newBuilder()
            .setBlockEnd(BlockEnd.newBuilder().setSubType(BlockEnd.SubType.ST_LIST)));
    return enterLoc;
  }

  private static <V1, V2> Location logPair(
      Pair<V1, V2> pair, StackTraceElement locationOfLogEvent, @Nullable Sight sight) {
    Location enterLoc =
        sight.enterBlock(
            "tuple",
            locationOfLogEvent,
            com.google.protos.sight.x.proto.Sight.Object.newBuilder()
                .setBlockStart(
                    BlockStart.newBuilder()
                        .setSubType(BlockStart.SubType.ST_LIST)
                        .setList(
                            ListStart.newBuilder().setSubType(ListStart.SubType.ST_HOMOGENEOUS))));
    log(pair.getFirst(), locationOfLogEvent, sight);
    log(pair.getSecond(), locationOfLogEvent, sight);
    sight.exitBlock(
        "tuple",
        locationOfLogEvent,
        com.google.protos.sight.x.proto.Sight.Object.newBuilder()
            .setBlockEnd(BlockEnd.newBuilder().setSubType(BlockEnd.SubType.ST_LIST)));
    return enterLoc;
  }

  private static <KeyT, ValT> Location logMap(
      Map<KeyT, ValT> map, StackTraceElement locationOfLogEvent, @Nullable Sight sight) {
    Location enterLoc =
        sight.enterBlock(
            "map",
            locationOfLogEvent,
            com.google.protos.sight.x.proto.Sight.Object.newBuilder()
                .setBlockStart(
                    BlockStart.newBuilder()
                        .setSubType(BlockStart.SubType.ST_LIST)
                        .setList(ListStart.newBuilder().setSubType(ListStart.SubType.ST_MAP))));
    for (Map.Entry<KeyT, ValT> e : map.entrySet()) {
      sight.enterBlock(
          "map.entry",
          locationOfLogEvent,
          com.google.protos.sight.x.proto.Sight.Object.newBuilder()
              .setBlockStart(
                  BlockStart.newBuilder()
                      .setSubType(BlockStart.SubType.ST_LIST)
                      .setList(ListStart.newBuilder().setSubType(ListStart.SubType.ST_MAP_ENTRY))));
      log(e.getKey(), locationOfLogEvent, sight);
      log(e.getValue(), locationOfLogEvent, sight);
      sight.exitBlock(
          "map.entry",
          locationOfLogEvent,
          com.google.protos.sight.x.proto.Sight.Object.newBuilder()
              .setBlockEnd(BlockEnd.newBuilder().setSubType(BlockEnd.SubType.ST_LIST)));
    }
    sight.exitBlock(
        "map",
        locationOfLogEvent,
        com.google.protos.sight.x.proto.Sight.Object.newBuilder()
            .setBlockEnd(BlockEnd.newBuilder().setSubType(BlockEnd.SubType.ST_LIST)));
    return enterLoc;
  }

  private static <T> Location logDict(
      Map<String, T> dict, StackTraceElement locationOfLogEvent, @Nullable Sight sight) {
    Location enterLoc =
        sight.enterBlock(
            "dict",
            locationOfLogEvent,
            com.google.protos.sight.x.proto.Sight.Object.newBuilder()
                .setBlockStart(
                    BlockStart.newBuilder()
                        .setSubType(BlockStart.SubType.ST_LIST)
                        .setList(ListStart.newBuilder().setSubType(ListStart.SubType.ST_DICT))));
    for (Map.Entry<String, T> e : dict.entrySet()) {
      log(e.getKey(), e.getValue(), locationOfLogEvent, sight);
    }
    sight.exitBlock(
        "dict",
        locationOfLogEvent,
        com.google.protos.sight.x.proto.Sight.Object.newBuilder()
            .setBlockEnd(BlockEnd.newBuilder().setSubType(BlockEnd.SubType.ST_LIST)));
    return enterLoc;
  }

  @ResultIgnorabilityUnspecified
  public static <T> Location logTable(
      List<String> keyColumns,
      List<String> valueColumns,
      List<double[]> rows,
      @Nullable Sight sight) {
    return logTable(keyColumns, valueColumns, rows, Sight.getCallerStackTraceElement(), sight);
  }

  private static <T> Location logTable(
      List<String> keyColumns,
      List<String> valueColumns,
      List<double[]> rows,
      StackTraceElement locationOfLogEvent,
      @Nullable Sight sight) {
    Location enterLoc =
        sight.enterBlock(
            "table",
            locationOfLogEvent,
            com.google.protos.sight.x.proto.Sight.Object.newBuilder()
                .setBlockStart(BlockStart.newBuilder().setSubType(BlockStart.SubType.ST_TABLE)));
    log(keyColumns, locationOfLogEvent, sight);
    log(valueColumns, locationOfLogEvent, sight);
    log(rows, locationOfLogEvent, sight);
    sight.exitBlock(
        "table",
        locationOfLogEvent,
        com.google.protos.sight.x.proto.Sight.Object.newBuilder()
            .setBlockEnd(BlockEnd.newBuilder().setSubType(BlockEnd.SubType.ST_TABLE)));
    return enterLoc;
  }

  private Collections() {}
}

