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

import static com.google.common.truth.Truth.assertThat;
import static com.google.common.truth.extensions.proto.ProtoTruth.assertThat;
import static java.nio.charset.StandardCharsets.UTF_8;

import com.google.analysis.dremel.core.capacitor.CapacitorException;
import com.google.analysis.dremel.core.capacitor.RecordReader;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.flogger.GoogleLogger;
import com.google.common.time.testing.FakeTimeSource;
import com.google.protobuf.Descriptors.FieldDescriptor;
import com.google.protobuf.ExtensionRegistry;
import com.google.protos.sight.x.proto.Sight.BlockEnd;
import com.google.protos.sight.x.proto.Sight.BlockStart;
import com.google.protos.sight.x.proto.Sight.ListStart;
import com.google.protos.sight.x.proto.Sight.Object.Metrics;
import com.google.protos.sight.x.proto.Sight.Object.Order;
import com.google.protos.sight.x.proto.Sight.Object.SubType;
import com.google.protos.sight.x.proto.Sight.Params;
import com.google.protos.sight.x.proto.Sight.Value;
import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.time.Duration;
import java.time.Instant;
import java.util.ArrayList;
import java.util.List;
import org.apache.commons.io.FileUtils;
import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.TemporaryFolder;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

@RunWith(JUnit4.class)
public final class SightCollectionsTest {
  @Rule public final TemporaryFolder tempFolder = new TemporaryFolder();
  private static final GoogleLogger logger = GoogleLogger.forEnclosingClass();

  private static final ImmutableList<FieldDescriptor> IGNORABLE_FIELDS =
      ImmutableList.of(
          com.google.protos.sight.x.proto.Sight.Object.getDescriptor()
              .findFieldByNumber(com.google.protos.sight.x.proto.Sight.Object.METRICS_FIELD_NUMBER),
          Metrics.getDescriptor()
              .findFieldByNumber(Metrics.PROCESS_FREE_SWAP_SPACE_BYTES_FIELD_NUMBER),
          Metrics.getDescriptor()
              .findFieldByNumber(Metrics.PROCESS_TOTAL_SWAP_SPACE_BYTES_FIELD_NUMBER));

  private static List<com.google.protos.sight.x.proto.Sight.Object> loadLog(String filePath)
      throws CapacitorException, IOException {
    List<com.google.protos.sight.x.proto.Sight.Object> logObjects = new ArrayList<>();
    try (RecordReader reader = new RecordReader(filePath, Duration.ofMinutes(10).getSeconds())) {
      for (int i = 0; i < reader.numRecords(); ++i) {
        byte[] bytes = reader.read();
        if (bytes == null) {
          logger.atSevere().log("Unable to read record %d from file '%s'.", i, filePath);
          throw new IOException("Failed to read record");
        }
        logObjects.add(
            com.google.protos.sight.x.proto.Sight.Object.parseFrom(
                bytes, ExtensionRegistry.getGeneratedRegistry()));
      }
    }
    return logObjects;
  }

  @Test
  public void createSight() {
    // SET-UP
    Sight sight = SightImpl.create();

    assertThat(sight).isNotNull();
  }

  @Test
  public void logNamedValueToTextFile() throws CapacitorException, IOException {
    // SET-UP
    Path testDir =
        Files.createDirectories(
            Path.of(tempFolder.newFolder("logNamedValueToTextFile").getAbsolutePath()));
    String logFilePath;
    try (SightImpl sight =
        SightImpl.create(
            Params.newBuilder()
                .setLocal(true)
                .setTextOutput(true)
                .setLogDirPath(testDir.toString())
                .build(),
            /* timeSource= */ FakeTimeSource.create()
                .setNow(Instant.ofEpochMilli(1))
                .setAutoAdvance(Duration.ofNanos(1)))) {
      logFilePath = sight.getTextLogFilePath().toString();
      // ACT
      Collections.log("name", Integer.valueOf(123), sight);
    }

    // ASSERT
    String attrValues =
        "| class=com.google.googlex.cortex.sight.SightCollectionsTest,function=logNamedValueToTextFile,"
            + "runStartTime=1969-12-31T16:00:00.001,taskBNS=";
    assertThat(FileUtils.readFileToString(new File(logFilePath), UTF_8))
        .isEqualTo(String.format("name<<<%s\n" + "123\n" + "name>>>>%s\n", attrValues, attrValues));
  }

  @Test
  public void logNamedValueToCapacitorFile() throws CapacitorException, IOException {
    // SET-UP
    Path testDir =
        Files.createDirectories(
            Path.of(tempFolder.newFolder("logNamedValueToCapacitorFile").getAbsolutePath()));
    String logFilePath;
    StackTraceElement baseEventLoc = Thread.currentThread().getStackTrace()[1];
    try (SightImpl sight =
        SightImpl.create(
            Params.newBuilder()
                .setLocal(true)
                .setCapacitorOutput(true)
                .setLogDirPath(testDir.toString())
                .build(),
            /* timeSource= */ FakeTimeSource.create()
                .setNow(Instant.ofEpochMilli(1))
                .setAutoAdvance(Duration.ofNanos(1)))) {
      logFilePath = sight.getCapacitorLogFilePath().toString();
      // ACT
      Collections.log("name", Integer.valueOf(123), sight);
    }

    // ASSERT
    ImmutableList<com.google.protos.sight.x.proto.Sight.Attribute> blockAttrs =
        ImmutableList.of(
            com.google.protos.sight.x.proto.Sight.Attribute.newBuilder()
                .setKey("class")
                .setValue("com.google.googlex.cortex.sight.SightCollectionsTest")
                .build(),
            com.google.protos.sight.x.proto.Sight.Attribute.newBuilder()
                .setKey("function")
                .setValue("logNamedValueToCapacitorFile")
                .build(),
            com.google.protos.sight.x.proto.Sight.Attribute.newBuilder()
                .setKey("runStartTime")
                .setValue("1969-12-31T16:00:00.001")
                .build(),
            com.google.protos.sight.x.proto.Sight.Attribute.newBuilder()
                .setKey("taskBNS")
                .setValue("")
                .build());
    assertThat(loadLog(logFilePath))
        .ignoringFieldDescriptors(IGNORABLE_FIELDS)
        .containsExactly(
            com.google.protos.sight.x.proto.Sight.Object.newBuilder()
                .setLocation("0000000000")
                .setIndex(0)
                .addAncestorStartLocation("0000000000")
                .setFile("SightCollectionsTest.java")
                .setLine(baseEventLoc.getLineNumber() + 13)
                .setFunc("logNamedValueToCapacitorFile")
                .addAllAttribute(blockAttrs)
                .setSubType(SubType.ST_BLOCK_START)
                .setBlockStart(
                    BlockStart.newBuilder()
                        .setLabel("name")
                        .setSubType(BlockStart.SubType.ST_NAMED_VALUE))
                .setOrder(Order.newBuilder().setTimestampNs(1000001))
                .build(),
            com.google.protos.sight.x.proto.Sight.Object.newBuilder()
                .setLocation("0000000000:0000000000")
                .setIndex(1)
                .addAncestorStartLocation("0000000000")
                .addAncestorStartLocation("0000000000:0000000000")
                .setFile("SightCollectionsTest.java")
                .setLine(baseEventLoc.getLineNumber() + 13)
                .setFunc("logNamedValueToCapacitorFile")
                .addAllAttribute(blockAttrs)
                .setSubType(SubType.ST_VALUE)
                .setValue(Value.newBuilder().setSubType(Value.SubType.ST_INT64).setInt64Value(123))
                .setOrder(Order.newBuilder().setTimestampNs(1000002))
                .build(),
            com.google.protos.sight.x.proto.Sight.Object.newBuilder()
                .setLocation("0000000001")
                .setIndex(2)
                .addAncestorStartLocation("0000000001")
                .setFile("SightCollectionsTest.java")
                .setLine(baseEventLoc.getLineNumber() + 13)
                .setFunc("logNamedValueToCapacitorFile")
                .addAllAttribute(blockAttrs)
                .setSubType(SubType.ST_BLOCK_END)
                .setBlockEnd(
                    BlockEnd.newBuilder()
                        .setLabel("name")
                        .setLocationOfBlockStart("0000000000")
                        .setNumDirectContents(1)
                        .setNumTransitiveContents(1)
                        .setSubType(BlockEnd.SubType.ST_NAMED_VALUE)
                        .setMetrics(BlockEnd.Metrics.newBuilder().setElapsedTimeNs(2)))
                .setOrder(Order.newBuilder().setTimestampNs(1000003))
                .build());
  }

  @Test
  public void logStringListToTextFile() throws CapacitorException, IOException {
    // SET-UP
    Path testDir =
        Files.createDirectories(
            Path.of(tempFolder.newFolder("logStringListToTextFile").getAbsolutePath()));
    String logFilePath;
    try (SightImpl sight =
        SightImpl.create(
            Params.newBuilder()
                .setLocal(true)
                .setTextOutput(true)
                .setLogDirPath(testDir.toString())
                .build(),
            /* timeSource= */ FakeTimeSource.create()
                .setNow(Instant.ofEpochMilli(1))
                .setAutoAdvance(Duration.ofNanos(1)))) {
      logFilePath = sight.getTextLogFilePath().toString();
      // ACT
      Collections.log(ImmutableList.of("foo", "bar", "baz"), sight);
    }
    // ASSERT
    String attrValues =
        "| class=com.google.googlex.cortex.sight.SightCollectionsTest,function=logStringListToTextFile,"
            + "runStartTime=1969-12-31T16:00:00.001,taskBNS=";
    assertThat(FileUtils.readFileToString(new File(logFilePath), UTF_8))
        .isEqualTo(
            String.format(
                "list<<<%s\n" + "foo\n" + "bar\n" + "baz\n" + "list>>>>%s\n",
                attrValues, attrValues));
  }

  @Test
  public void logStringListToCapacitorFile() throws CapacitorException, IOException {
    // SET-UP
    Path testDir =
        Files.createDirectories(
            Path.of(tempFolder.newFolder("logStringListToCapacitorFile").getAbsolutePath()));
    String logFilePath;
    StackTraceElement baseEventLoc = Thread.currentThread().getStackTrace()[1];
    try (SightImpl sight =
        SightImpl.create(
            Params.newBuilder()
                .setLocal(true)
                .setCapacitorOutput(true)
                .setLogDirPath(testDir.toString())
                .build(),
            /* timeSource= */ FakeTimeSource.create()
                .setNow(Instant.ofEpochMilli(1))
                .setAutoAdvance(Duration.ofNanos(1)))) {
      logFilePath = sight.getCapacitorLogFilePath().toString();
      // ACT
      Collections.log(ImmutableList.of("foo", "bar", "baz"), sight);
    }

    // ASSERT
    ImmutableList<com.google.protos.sight.x.proto.Sight.Attribute> blockAttrs =
        ImmutableList.of(
            com.google.protos.sight.x.proto.Sight.Attribute.newBuilder()
                .setKey("class")
                .setValue("com.google.googlex.cortex.sight.SightCollectionsTest")
                .build(),
            com.google.protos.sight.x.proto.Sight.Attribute.newBuilder()
                .setKey("function")
                .setValue("logStringListToCapacitorFile")
                .build(),
            com.google.protos.sight.x.proto.Sight.Attribute.newBuilder()
                .setKey("runStartTime")
                .setValue("1969-12-31T16:00:00.001")
                .build(),
            com.google.protos.sight.x.proto.Sight.Attribute.newBuilder()
                .setKey("taskBNS")
                .setValue("")
                .build());
    assertThat(loadLog(logFilePath))
        .ignoringFieldDescriptors(IGNORABLE_FIELDS)
        .containsExactly(
            com.google.protos.sight.x.proto.Sight.Object.newBuilder()
                .setLocation("0000000000")
                .setIndex(0)
                .addAncestorStartLocation("0000000000")
                .setFile("SightCollectionsTest.java")
                .setLine(baseEventLoc.getLineNumber() + 13)
                .setFunc("logStringListToCapacitorFile")
                .addAllAttribute(blockAttrs)
                .setSubType(SubType.ST_BLOCK_START)
                .setBlockStart(
                    BlockStart.newBuilder()
                        .setLabel("list")
                        .setSubType(BlockStart.SubType.ST_LIST)
                        .setList(
                            ListStart.newBuilder().setSubType(ListStart.SubType.ST_HOMOGENEOUS)))
                .setOrder(Order.newBuilder().setTimestampNs(1000001))
                .build(),
            com.google.protos.sight.x.proto.Sight.Object.newBuilder()
                .setLocation("0000000000:0000000000")
                .setIndex(1)
                .addAncestorStartLocation("0000000000")
                .addAncestorStartLocation("0000000000:0000000000")
                .setFile("SightCollectionsTest.java")
                .setLine(baseEventLoc.getLineNumber() + 13)
                .setFunc("logStringListToCapacitorFile")
                .addAllAttribute(blockAttrs)
                .setSubType(SubType.ST_VALUE)
                .setValue(
                    Value.newBuilder().setSubType(Value.SubType.ST_STRING).setStringValue("foo"))
                .setOrder(Order.newBuilder().setTimestampNs(1000002))
                .build(),
            com.google.protos.sight.x.proto.Sight.Object.newBuilder()
                .setLocation("0000000000:0000000001")
                .setIndex(2)
                .addAncestorStartLocation("0000000000")
                .addAncestorStartLocation("0000000000:0000000001")
                .setFile("SightCollectionsTest.java")
                .setLine(baseEventLoc.getLineNumber() + 13)
                .setFunc("logStringListToCapacitorFile")
                .addAllAttribute(blockAttrs)
                .setSubType(SubType.ST_VALUE)
                .setValue(
                    Value.newBuilder().setSubType(Value.SubType.ST_STRING).setStringValue("bar"))
                .setOrder(Order.newBuilder().setTimestampNs(1000003))
                .build(),
            com.google.protos.sight.x.proto.Sight.Object.newBuilder()
                .setLocation("0000000000:0000000002")
                .setIndex(3)
                .addAncestorStartLocation("0000000000")
                .addAncestorStartLocation("0000000000:0000000002")
                .setFile("SightCollectionsTest.java")
                .setLine(baseEventLoc.getLineNumber() + 13)
                .setFunc("logStringListToCapacitorFile")
                .addAllAttribute(blockAttrs)
                .setSubType(SubType.ST_VALUE)
                .setValue(
                    Value.newBuilder().setSubType(Value.SubType.ST_STRING).setStringValue("baz"))
                .setOrder(Order.newBuilder().setTimestampNs(1000004))
                .build(),
            com.google.protos.sight.x.proto.Sight.Object.newBuilder()
                .setLocation("0000000001")
                .setIndex(4)
                .addAncestorStartLocation("0000000001")
                .setFile("SightCollectionsTest.java")
                .setLine(baseEventLoc.getLineNumber() + 13)
                .setFunc("logStringListToCapacitorFile")
                .addAllAttribute(blockAttrs)
                .setSubType(SubType.ST_BLOCK_END)
                .setBlockEnd(
                    BlockEnd.newBuilder()
                        .setLabel("list")
                        .setLocationOfBlockStart("0000000000")
                        .setNumDirectContents(3)
                        .setNumTransitiveContents(3)
                        .setSubType(BlockEnd.SubType.ST_LIST)
                        .setMetrics(BlockEnd.Metrics.newBuilder().setElapsedTimeNs(4)))
                .setOrder(Order.newBuilder().setTimestampNs(1000005))
                .build());
  }

  @Test
  public void logMapIntToStringListToTextFile() throws CapacitorException, IOException {
    // SET-UP
    Path testDir =
        Files.createDirectories(
            Path.of(tempFolder.newFolder("logMapIntToStringListToTextFile").getAbsolutePath()));
    String logFilePath;
    try (SightImpl sight =
        SightImpl.create(
            Params.newBuilder()
                .setLocal(true)
                .setTextOutput(true)
                .setLogDirPath(testDir.toString())
                .build(),
            /* timeSource= */ FakeTimeSource.create()
                .setNow(Instant.ofEpochMilli(1))
                .setAutoAdvance(Duration.ofNanos(1)))) {
      logFilePath = sight.getTextLogFilePath().toString();
      // ACT
      Collections.log(ImmutableMap.of(1, ImmutableList.of("foo", "bar")), sight);
    }

    // ASSERT
    String attrValues =
        "| class=com.google.googlex.cortex.sight.SightCollectionsTest,function=logMapIntToStringListToTextFile,"
            + "runStartTime=1969-12-31T16:00:00.001,taskBNS=";
    assertThat(FileUtils.readFileToString(new File(logFilePath), UTF_8))
        .isEqualTo(
            String.format(
                "map<<<%s\n"
                    + "map: map.entry<<<%s\n"
                    + "1\n"
                    + "map: map.entry: list<<<%s\n"
                    + "foo\n"
                    + "bar\n"
                    + "map: map.entry: list>>>>%s\n"
                    + "map: map.entry>>>>%s\n"
                    + "map>>>>%s\n",
                attrValues, attrValues, attrValues, attrValues, attrValues, attrValues));
  }

  @Test
  public void logMapIntToStringListToCapacitorFile() throws CapacitorException, IOException {
    // SET-UP
    Path testDir =
        Files.createDirectories(
            Path.of(
                tempFolder.newFolder("logMapIntToStringListToCapacitorFile").getAbsolutePath()));
    String logFilePath;
    StackTraceElement baseEventLoc = Thread.currentThread().getStackTrace()[1];
    try (SightImpl sight =
        SightImpl.create(
            Params.newBuilder()
                .setLocal(true)
                .setCapacitorOutput(true)
                .setLogDirPath(testDir.toString())
                .build(),
            /* timeSource= */ FakeTimeSource.create()
                .setNow(Instant.ofEpochMilli(1))
                .setAutoAdvance(Duration.ofNanos(1)))) {
      logFilePath = sight.getCapacitorLogFilePath().toString();
      // ACT
      Collections.log(
          ImmutableMap.of(1, ImmutableList.of("foo", "bar"), 9, ImmutableList.of("x")), sight);
    }

    // ASSERT
    ImmutableList<com.google.protos.sight.x.proto.Sight.Attribute> blockAttrs =
        ImmutableList.of(
            com.google.protos.sight.x.proto.Sight.Attribute.newBuilder()
                .setKey("class")
                .setValue("com.google.googlex.cortex.sight.SightCollectionsTest")
                .build(),
            com.google.protos.sight.x.proto.Sight.Attribute.newBuilder()
                .setKey("function")
                .setValue("logMapIntToStringListToCapacitorFile")
                .build(),
            com.google.protos.sight.x.proto.Sight.Attribute.newBuilder()
                .setKey("runStartTime")
                .setValue("1969-12-31T16:00:00.001")
                .build(),
            com.google.protos.sight.x.proto.Sight.Attribute.newBuilder()
                .setKey("taskBNS")
                .setValue("")
                .build());
    assertThat(loadLog(logFilePath))
        .ignoringFieldDescriptors(IGNORABLE_FIELDS)
        .containsExactly(
            com.google.protos.sight.x.proto.Sight.Object.newBuilder()
                .setLocation("0000000000")
                .setIndex(0)
                .addAncestorStartLocation("0000000000")
                .setFile("SightCollectionsTest.java")
                .setLine(baseEventLoc.getLineNumber() + 13)
                .setFunc("logMapIntToStringListToCapacitorFile")
                .addAllAttribute(blockAttrs)
                .setSubType(SubType.ST_BLOCK_START)
                .setBlockStart(
                    BlockStart.newBuilder()
                        .setLabel("map")
                        .setSubType(BlockStart.SubType.ST_LIST)
                        .setList(ListStart.newBuilder().setSubType(ListStart.SubType.ST_MAP)))
                .setOrder(Order.newBuilder().setTimestampNs(1000001))
                .build(),

            // 1->["foo", "bar"]
            com.google.protos.sight.x.proto.Sight.Object.newBuilder()
                .setLocation("0000000000:0000000000")
                .setIndex(1)
                .addAncestorStartLocation("0000000000")
                .addAncestorStartLocation("0000000000:0000000000")
                .setFile("SightCollectionsTest.java")
                .setLine(baseEventLoc.getLineNumber() + 13)
                .setFunc("logMapIntToStringListToCapacitorFile")
                .addAllAttribute(blockAttrs)
                .setSubType(SubType.ST_BLOCK_START)
                .setBlockStart(
                    BlockStart.newBuilder()
                        .setLabel("map.entry")
                        .setSubType(BlockStart.SubType.ST_LIST)
                        .setList(ListStart.newBuilder().setSubType(ListStart.SubType.ST_MAP_ENTRY)))
                .setOrder(Order.newBuilder().setTimestampNs(1000002))
                .build(),
            com.google.protos.sight.x.proto.Sight.Object.newBuilder()
                .setLocation("0000000000:0000000000:0000000000")
                .setIndex(2)
                .addAncestorStartLocation("0000000000")
                .addAncestorStartLocation("0000000000:0000000000")
                .addAncestorStartLocation("0000000000:0000000000:0000000000")
                .setFile("SightCollectionsTest.java")
                .setLine(baseEventLoc.getLineNumber() + 13)
                .setFunc("logMapIntToStringListToCapacitorFile")
                .addAllAttribute(blockAttrs)
                .setSubType(SubType.ST_VALUE)
                .setValue(Value.newBuilder().setSubType(Value.SubType.ST_INT64).setInt64Value(1))
                .setOrder(Order.newBuilder().setTimestampNs(1000003))
                .build(),

            // ["foo", "bar"]
            com.google.protos.sight.x.proto.Sight.Object.newBuilder()
                .setLocation("0000000000:0000000000:0000000001")
                .setIndex(3)
                .addAncestorStartLocation("0000000000")
                .addAncestorStartLocation("0000000000:0000000000")
                .addAncestorStartLocation("0000000000:0000000000:0000000001")
                .setFile("SightCollectionsTest.java")
                .setLine(baseEventLoc.getLineNumber() + 13)
                .setFunc("logMapIntToStringListToCapacitorFile")
                .addAllAttribute(blockAttrs)
                .setSubType(SubType.ST_BLOCK_START)
                .setBlockStart(
                    BlockStart.newBuilder()
                        .setLabel("list")
                        .setSubType(BlockStart.SubType.ST_LIST)
                        .setList(
                            ListStart.newBuilder().setSubType(ListStart.SubType.ST_HOMOGENEOUS)))
                .setOrder(Order.newBuilder().setTimestampNs(1000004))
                .build(),
            com.google.protos.sight.x.proto.Sight.Object.newBuilder()
                .setLocation("0000000000:0000000000:0000000001:0000000000")
                .setIndex(4)
                .addAncestorStartLocation("0000000000")
                .addAncestorStartLocation("0000000000:0000000000")
                .addAncestorStartLocation("0000000000:0000000000:0000000001")
                .addAncestorStartLocation("0000000000:0000000000:0000000001:0000000000")
                .setFile("SightCollectionsTest.java")
                .setLine(baseEventLoc.getLineNumber() + 13)
                .setFunc("logMapIntToStringListToCapacitorFile")
                .addAllAttribute(blockAttrs)
                .setSubType(SubType.ST_VALUE)
                .setValue(
                    Value.newBuilder().setSubType(Value.SubType.ST_STRING).setStringValue("foo"))
                .setOrder(Order.newBuilder().setTimestampNs(1000005))
                .build(),
            com.google.protos.sight.x.proto.Sight.Object.newBuilder()
                .setLocation("0000000000:0000000000:0000000001:0000000001")
                .setIndex(5)
                .addAncestorStartLocation("0000000000")
                .addAncestorStartLocation("0000000000:0000000000")
                .addAncestorStartLocation("0000000000:0000000000:0000000001")
                .addAncestorStartLocation("0000000000:0000000000:0000000001:0000000001")
                .setFile("SightCollectionsTest.java")
                .setLine(baseEventLoc.getLineNumber() + 13)
                .setFunc("logMapIntToStringListToCapacitorFile")
                .addAllAttribute(blockAttrs)
                .setSubType(SubType.ST_VALUE)
                .setValue(
                    Value.newBuilder().setSubType(Value.SubType.ST_STRING).setStringValue("bar"))
                .setOrder(Order.newBuilder().setTimestampNs(1000006))
                .build(),
            com.google.protos.sight.x.proto.Sight.Object.newBuilder()
                .setLocation("0000000000:0000000000:0000000002")
                .setIndex(6)
                .addAncestorStartLocation("0000000000")
                .addAncestorStartLocation("0000000000:0000000000")
                .addAncestorStartLocation("0000000000:0000000000:0000000002")
                .setFile("SightCollectionsTest.java")
                .setLine(baseEventLoc.getLineNumber() + 13)
                .setFunc("logMapIntToStringListToCapacitorFile")
                .addAllAttribute(blockAttrs)
                .setSubType(SubType.ST_BLOCK_END)
                .setBlockEnd(
                    BlockEnd.newBuilder()
                        .setLabel("list")
                        .setLocationOfBlockStart("0000000000:0000000000:0000000001")
                        .setNumDirectContents(2)
                        .setNumTransitiveContents(2)
                        .setSubType(BlockEnd.SubType.ST_LIST)
                        .setMetrics(BlockEnd.Metrics.newBuilder().setElapsedTimeNs(3)))
                .setOrder(Order.newBuilder().setTimestampNs(1000007))
                .build(),
            com.google.protos.sight.x.proto.Sight.Object.newBuilder()
                .setLocation("0000000000:0000000001")
                .setIndex(7)
                .addAncestorStartLocation("0000000000")
                .addAncestorStartLocation("0000000000:0000000001")
                .setFile("SightCollectionsTest.java")
                .setLine(baseEventLoc.getLineNumber() + 13)
                .setFunc("logMapIntToStringListToCapacitorFile")
                .addAllAttribute(blockAttrs)
                .setSubType(SubType.ST_BLOCK_END)
                .setBlockEnd(
                    BlockEnd.newBuilder()
                        .setLabel("map.entry")
                        .setLocationOfBlockStart("0000000000:0000000000")
                        .setNumDirectContents(2)
                        .setNumTransitiveContents(5)
                        .setSubType(BlockEnd.SubType.ST_LIST)
                        .setMetrics(BlockEnd.Metrics.newBuilder().setElapsedTimeNs(6)))
                .setOrder(Order.newBuilder().setTimestampNs(1000008))
                .build(),

            // 9->["x"]
            com.google.protos.sight.x.proto.Sight.Object.newBuilder()
                .setLocation("0000000000:0000000002")
                .setIndex(8)
                .addAncestorStartLocation("0000000000")
                .addAncestorStartLocation("0000000000:0000000002")
                .setFile("SightCollectionsTest.java")
                .setLine(baseEventLoc.getLineNumber() + 13)
                .setFunc("logMapIntToStringListToCapacitorFile")
                .addAllAttribute(blockAttrs)
                .setSubType(SubType.ST_BLOCK_START)
                .setBlockStart(
                    BlockStart.newBuilder()
                        .setLabel("map.entry")
                        .setSubType(BlockStart.SubType.ST_LIST)
                        .setList(ListStart.newBuilder().setSubType(ListStart.SubType.ST_MAP_ENTRY)))
                .setOrder(Order.newBuilder().setTimestampNs(1000009))
                .build(),
            com.google.protos.sight.x.proto.Sight.Object.newBuilder()
                .setLocation("0000000000:0000000002:0000000000")
                .setIndex(9)
                .addAncestorStartLocation("0000000000")
                .addAncestorStartLocation("0000000000:0000000002")
                .addAncestorStartLocation("0000000000:0000000002:0000000000")
                .setFile("SightCollectionsTest.java")
                .setLine(baseEventLoc.getLineNumber() + 13)
                .setFunc("logMapIntToStringListToCapacitorFile")
                .addAllAttribute(blockAttrs)
                .setSubType(SubType.ST_VALUE)
                .setValue(Value.newBuilder().setSubType(Value.SubType.ST_INT64).setInt64Value(9))
                .setOrder(Order.newBuilder().setTimestampNs(1000010))
                .build(),

            // ["x"]
            com.google.protos.sight.x.proto.Sight.Object.newBuilder()
                .setLocation("0000000000:0000000002:0000000001")
                .setIndex(10)
                .addAncestorStartLocation("0000000000")
                .addAncestorStartLocation("0000000000:0000000002")
                .addAncestorStartLocation("0000000000:0000000002:0000000001")
                .setFile("SightCollectionsTest.java")
                .setLine(baseEventLoc.getLineNumber() + 13)
                .setFunc("logMapIntToStringListToCapacitorFile")
                .addAllAttribute(blockAttrs)
                .setSubType(SubType.ST_BLOCK_START)
                .setBlockStart(
                    BlockStart.newBuilder()
                        .setLabel("list")
                        .setSubType(BlockStart.SubType.ST_LIST)
                        .setList(
                            ListStart.newBuilder().setSubType(ListStart.SubType.ST_HOMOGENEOUS)))
                .setOrder(Order.newBuilder().setTimestampNs(1000011))
                .build(),
            com.google.protos.sight.x.proto.Sight.Object.newBuilder()
                .setLocation("0000000000:0000000002:0000000001:0000000000")
                .setIndex(11)
                .addAncestorStartLocation("0000000000")
                .addAncestorStartLocation("0000000000:0000000002")
                .addAncestorStartLocation("0000000000:0000000002:0000000001")
                .addAncestorStartLocation("0000000000:0000000002:0000000001:0000000000")
                .setFile("SightCollectionsTest.java")
                .setLine(baseEventLoc.getLineNumber() + 13)
                .setFunc("logMapIntToStringListToCapacitorFile")
                .addAllAttribute(blockAttrs)
                .setSubType(SubType.ST_VALUE)
                .setValue(
                    Value.newBuilder().setSubType(Value.SubType.ST_STRING).setStringValue("x"))
                .setOrder(Order.newBuilder().setTimestampNs(1000012))
                .build(),
            com.google.protos.sight.x.proto.Sight.Object.newBuilder()
                .setLocation("0000000000:0000000002:0000000002")
                .setIndex(12)
                .addAncestorStartLocation("0000000000")
                .addAncestorStartLocation("0000000000:0000000002")
                .addAncestorStartLocation("0000000000:0000000002:0000000002")
                .setFile("SightCollectionsTest.java")
                .setLine(baseEventLoc.getLineNumber() + 13)
                .setFunc("logMapIntToStringListToCapacitorFile")
                .addAllAttribute(blockAttrs)
                .setSubType(SubType.ST_BLOCK_END)
                .setBlockEnd(
                    BlockEnd.newBuilder()
                        .setLabel("list")
                        .setLocationOfBlockStart("0000000000:0000000002:0000000001")
                        .setNumDirectContents(1)
                        .setNumTransitiveContents(1)
                        .setSubType(BlockEnd.SubType.ST_LIST)
                        .setMetrics(BlockEnd.Metrics.newBuilder().setElapsedTimeNs(2)))
                .setOrder(Order.newBuilder().setTimestampNs(1000013))
                .build(),
            com.google.protos.sight.x.proto.Sight.Object.newBuilder()
                .setLocation("0000000000:0000000003")
                .setIndex(13)
                .addAncestorStartLocation("0000000000")
                .addAncestorStartLocation("0000000000:0000000003")
                .setFile("SightCollectionsTest.java")
                .setLine(baseEventLoc.getLineNumber() + 13)
                .setFunc("logMapIntToStringListToCapacitorFile")
                .addAllAttribute(blockAttrs)
                .setSubType(SubType.ST_BLOCK_END)
                .setBlockEnd(
                    BlockEnd.newBuilder()
                        .setLabel("map.entry")
                        .setLocationOfBlockStart("0000000000:0000000002")
                        .setNumDirectContents(2)
                        .setNumTransitiveContents(4)
                        .setSubType(BlockEnd.SubType.ST_LIST)
                        .setMetrics(BlockEnd.Metrics.newBuilder().setElapsedTimeNs(5)))
                .setOrder(Order.newBuilder().setTimestampNs(1000014))
                .build(),
            com.google.protos.sight.x.proto.Sight.Object.newBuilder()
                .setLocation("0000000001")
                .setIndex(14)
                .addAncestorStartLocation("0000000001")
                .setFile("SightCollectionsTest.java")
                .setLine(baseEventLoc.getLineNumber() + 13)
                .setFunc("logMapIntToStringListToCapacitorFile")
                .addAllAttribute(blockAttrs)
                .setSubType(SubType.ST_BLOCK_END)
                .setBlockEnd(
                    BlockEnd.newBuilder()
                        .setLabel("map")
                        .setLocationOfBlockStart("0000000000")
                        .setNumDirectContents(2)
                        .setNumTransitiveContents(13)
                        .setSubType(BlockEnd.SubType.ST_LIST)
                        .setMetrics(BlockEnd.Metrics.newBuilder().setElapsedTimeNs(14)))
                .setOrder(Order.newBuilder().setTimestampNs(1000015))
                .build());
  }
}
