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

package com.google.googlex.cortex.sight.widgets.protobuf;

import com.google.errorprone.annotations.ResultIgnorabilityUnspecified;
import com.google.googlex.cortex.sight.Collections;
import com.google.googlex.cortex.sight.Location;
import com.google.googlex.cortex.sight.Sight;
import com.google.googlex.cortex.sight.SightLoggable;
import com.google.protobuf.Descriptors.FieldDescriptor;
import com.google.protobuf.MessageOrBuilder;
import com.google.protos.sight.x.proto.Sight.BlockEnd;
import com.google.protos.sight.x.proto.Sight.BlockStart;
import com.google.protos.sight.x.proto.Sight.ListStart;
import java.util.List;
import java.util.Map;

/**
 * Class that makes it possible to log protobufs and containers via Sight either by wrapping proto
 * messages with a SightLoggable interface or via static logging methods.
 */
public final class Proto implements SightLoggable {
  private MessageOrBuilder mesg;

  public static Proto of(MessageOrBuilder mesg) {
    Proto proto = new Proto();
    proto.mesg = mesg;
    return proto;
  }

  // Static methods for logging proto {@code mesg} without wrapping.
  @ResultIgnorabilityUnspecified
  public static Location log(MessageOrBuilder mesg, Sight sight) {
    return Proto.of(mesg).log(sight);
  }

  @ResultIgnorabilityUnspecified
  @Override
  public Location log(Sight sight) {
    return log(Sight.getCallerStackTraceElement(), sight);
  }

  @ResultIgnorabilityUnspecified
  @Override
  public Location log(StackTraceElement locationOfLogEvent, Sight sight) {
    Location enterLoc =
        sight.enterBlock(
            mesg.getDescriptorForType().getName(),
            locationOfLogEvent,
            com.google.protos.sight.x.proto.Sight.Object.newBuilder()
                .setBlockStart(
                    BlockStart.newBuilder()
                        .setSubType(BlockStart.SubType.ST_LIST)
                        .setList(ListStart.newBuilder().setSubType(ListStart.SubType.ST_DICT))));

    for (Map.Entry<FieldDescriptor, Object> field : mesg.getAllFields().entrySet()) {
      String fieldName = field.getKey().getName();

      if (field.getKey().isMapField()) {
        @SuppressWarnings("unchecked") // Proto API ensures that value of map fields must
        // be a list.
        List<Object> mapEntries = (List<Object>) field.getValue();
        logMapField(field.getKey(), mapEntries, locationOfLogEvent, sight);
        continue;
      }
      if (field.getKey().isRepeated()) {
        @SuppressWarnings("unchecked") // Proto API ensures that value of map fields must
        // be a list.
        List<Object> repeatedField = (List<Object>) field.getValue();
        logRepeatedField(field.getKey(), repeatedField, locationOfLogEvent, sight);
        continue;
      }
      Collections.log(fieldName, field.getValue(), locationOfLogEvent, sight);
    }

    sight.exitBlock(
        mesg.getDescriptorForType().getName(),
        locationOfLogEvent,
        com.google.protos.sight.x.proto.Sight.Object.newBuilder()
            .setBlockEnd(BlockEnd.newBuilder().setSubType(BlockEnd.SubType.ST_LIST)));

    return enterLoc;
  }

  private static void logMapField(
      FieldDescriptor desc,
      List<Object> mapEntries,
      StackTraceElement locationOfLogEvent,
      Sight sight) {
    sight.enterBlock(
        desc.getName(),
        locationOfLogEvent,
        com.google.protos.sight.x.proto.Sight.Object.newBuilder()
            .setBlockStart(
                BlockStart.newBuilder()
                    .setSubType(BlockStart.SubType.ST_LIST)
                    .setList(ListStart.newBuilder().setSubType(ListStart.SubType.ST_DICT))));

    for (Object entry : mapEntries) {
      String mapKey = null;
      Object mapValue = null;
      FieldDescriptor mapValueDesc = null;
      for (Map.Entry<FieldDescriptor, Object> mapField :
          ((MessageOrBuilder) entry).getAllFields().entrySet()) {
        if (mapField.getKey().getName().equals("key")) {
          mapKey = (String) mapField.getValue();
        } else if (mapField.getKey().getName().equals("value")) {
          mapValue = mapField.getValue();
          mapValueDesc = mapField.getKey();
        }
      }

      sight.enterBlock(
          mapKey,
          locationOfLogEvent,
          com.google.protos.sight.x.proto.Sight.Object.newBuilder()
              .setBlockStart(
                  BlockStart.newBuilder().setSubType(BlockStart.SubType.ST_NAMED_VALUE)));
      if (mapValueDesc.getType() == FieldDescriptor.Type.MESSAGE) {
        MessageOrBuilder subMessage = (MessageOrBuilder) mapValue;
        Proto.of(subMessage).log(locationOfLogEvent, sight);
      } else {
        Collections.log(mapValue, locationOfLogEvent, sight);
      }
      sight.exitBlock(
          mapKey,
          locationOfLogEvent,
          com.google.protos.sight.x.proto.Sight.Object.newBuilder()
              .setBlockEnd(BlockEnd.newBuilder().setSubType(BlockEnd.SubType.ST_NAMED_VALUE)));
    }

    sight.exitBlock(
        desc.getName(),
        locationOfLogEvent,
        com.google.protos.sight.x.proto.Sight.Object.newBuilder()
            .setBlockEnd(BlockEnd.newBuilder().setSubType(BlockEnd.SubType.ST_LIST)));
  }

  private static void logRepeatedField(
      FieldDescriptor desc,
      List<Object> repeatedField,
      StackTraceElement locationOfLogEvent,
      Sight sight) {
    sight.enterBlock(
        "repeated",
        locationOfLogEvent,
        com.google.protos.sight.x.proto.Sight.Object.newBuilder()
            .setBlockStart(
                BlockStart.newBuilder()
                    .setSubType(BlockStart.SubType.ST_LIST)
                    .setList(ListStart.newBuilder().setSubType(ListStart.SubType.ST_HOMOGENEOUS))));
    for (int i = 0; i < repeatedField.size(); ++i) {
      if (desc.getType() == FieldDescriptor.Type.MESSAGE) {
        MessageOrBuilder curField = (MessageOrBuilder) repeatedField.get(i);
        Proto.of(curField).log(locationOfLogEvent, sight);
      } else {
        Collections.log(repeatedField.get(i), locationOfLogEvent, sight);
      }
    }
    sight.exitBlock(
        "repeated",
        locationOfLogEvent,
        com.google.protos.sight.x.proto.Sight.Object.newBuilder()
            .setBlockEnd(BlockEnd.newBuilder().setSubType(BlockEnd.SubType.ST_LIST)));
  }
}
