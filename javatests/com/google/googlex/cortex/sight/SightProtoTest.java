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
import com.google.common.flogger.GoogleLogger;
import com.google.common.time.testing.FakeTimeSource;
import com.google.googlex.cortex.sight.widgets.protobuf.Proto;
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
import com.google.protos.sight.x.proto.test.SightTestMesg.TestMessage;
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
public final class SightProtoTest {
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
  public void logMesgWithStrFieldToTextFile() throws CapacitorException, IOException {
    // SET-UP
    Path testDir =
        Files.createDirectories(
            Path.of(tempFolder.newFolder("logMesgWithStrFieldToTextFile").getAbsolutePath()));
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
      Proto.of(TestMessage.newBuilder().setStrVal("value").setIntVal(123).build()).log(sight);
    }

    // ASSERT
    String attrValues =
        "| class=com.google.googlex.cortex.sight.SightProtoTest,function=logMesgWithStrFieldToTextFile,"
            + "runStartTime=1969-12-31T16:00:00.001,taskBNS=";
    assertThat(FileUtils.readFileToString(new File(logFilePath), UTF_8))
        .isEqualTo(
            String.format(
                "TestMessage<<<%s\n"
                    + "TestMessage: str_val<<<%s\n"
                    + "value\n"
                    + "TestMessage: str_val>>>>%s\n"
                    + "TestMessage: int_val<<<%s\n"
                    + "123\n"
                    + "TestMessage: int_val>>>>%s\n"
                    + "TestMessage>>>>%s\n",
                attrValues, attrValues, attrValues, attrValues, attrValues, attrValues));
  }

  @Test
  public void logMesgWithStrFieldToCapacitorFile() throws CapacitorException, IOException {
    // SET-UP
    Path testDir =
        Files.createDirectories(
            Path.of(tempFolder.newFolder("logMesgWithStrFieldToCapacitorFile").getAbsolutePath()));
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
      Proto.of(TestMessage.newBuilder().setStrVal("value").setIntVal(123).build()).log(sight);
    }

    // ASSERT
    ImmutableList<com.google.protos.sight.x.proto.Sight.Attribute> blockAttrs =
        ImmutableList.of(
            com.google.protos.sight.x.proto.Sight.Attribute.newBuilder()
                .setKey("class")
                .setValue("com.google.googlex.cortex.sight.SightProtoTest")
                .build(),
            com.google.protos.sight.x.proto.Sight.Attribute.newBuilder()
                .setKey("function")
                .setValue("logMesgWithStrFieldToCapacitorFile")
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
                .setFile("SightProtoTest.java")
                .setLine(baseEventLoc.getLineNumber() + 13)
                .setFunc("logMesgWithStrFieldToCapacitorFile")
                .addAllAttribute(blockAttrs)
                .setSubType(SubType.ST_BLOCK_START)
                .setBlockStart(
                    BlockStart.newBuilder()
                        .setLabel("TestMessage")
                        .setSubType(BlockStart.SubType.ST_LIST)
                        .setList(ListStart.newBuilder().setSubType(ListStart.SubType.ST_DICT)))
                .setOrder(Order.newBuilder().setTimestampNs(1000001))
                .build(),
            com.google.protos.sight.x.proto.Sight.Object.newBuilder()
                .setLocation("0000000000:0000000000")
                .setIndex(1)
                .addAncestorStartLocation("0000000000")
                .addAncestorStartLocation("0000000000:0000000000")
                .setFile("SightProtoTest.java")
                .setLine(baseEventLoc.getLineNumber() + 13)
                .setFunc("logMesgWithStrFieldToCapacitorFile")
                .addAllAttribute(blockAttrs)
                .setSubType(SubType.ST_BLOCK_START)
                .setBlockStart(
                    BlockStart.newBuilder()
                        .setLabel("str_val")
                        .setSubType(BlockStart.SubType.ST_NAMED_VALUE))
                .setOrder(Order.newBuilder().setTimestampNs(1000002))
                .build(),
            com.google.protos.sight.x.proto.Sight.Object.newBuilder()
                .setLocation("0000000000:0000000000:0000000000")
                .setIndex(2)
                .addAncestorStartLocation("0000000000")
                .addAncestorStartLocation("0000000000:0000000000")
                .addAncestorStartLocation("0000000000:0000000000:0000000000")
                .setFile("SightProtoTest.java")
                .setLine(baseEventLoc.getLineNumber() + 13)
                .setFunc("logMesgWithStrFieldToCapacitorFile")
                .addAllAttribute(blockAttrs)
                .setSubType(SubType.ST_VALUE)
                .setValue(
                    Value.newBuilder().setSubType(Value.SubType.ST_STRING).setStringValue("value"))
                .setOrder(Order.newBuilder().setTimestampNs(1000003))
                .build(),
            com.google.protos.sight.x.proto.Sight.Object.newBuilder()
                .setLocation("0000000000:0000000001")
                .setIndex(3)
                .addAncestorStartLocation("0000000000")
                .addAncestorStartLocation("0000000000:0000000001")
                .setFile("SightProtoTest.java")
                .setLine(baseEventLoc.getLineNumber() + 13)
                .setFunc("logMesgWithStrFieldToCapacitorFile")
                .addAllAttribute(blockAttrs)
                .setSubType(SubType.ST_BLOCK_END)
                .setBlockEnd(
                    BlockEnd.newBuilder()
                        .setLabel("str_val")
                        .setLocationOfBlockStart("0000000000:0000000000")
                        .setNumDirectContents(1)
                        .setNumTransitiveContents(1)
                        .setSubType(BlockEnd.SubType.ST_NAMED_VALUE)
                        .setMetrics(BlockEnd.Metrics.newBuilder().setElapsedTimeNs(2)))
                .setOrder(Order.newBuilder().setTimestampNs(1000004))
                .build(),
            com.google.protos.sight.x.proto.Sight.Object.newBuilder()
                .setLocation("0000000000:0000000002")
                .setIndex(4)
                .addAncestorStartLocation("0000000000")
                .addAncestorStartLocation("0000000000:0000000002")
                .setFile("SightProtoTest.java")
                .setLine(baseEventLoc.getLineNumber() + 13)
                .setFunc("logMesgWithStrFieldToCapacitorFile")
                .addAllAttribute(blockAttrs)
                .setSubType(SubType.ST_BLOCK_START)
                .setBlockStart(
                    BlockStart.newBuilder()
                        .setLabel("int_val")
                        .setSubType(BlockStart.SubType.ST_NAMED_VALUE))
                .setOrder(Order.newBuilder().setTimestampNs(1000005))
                .build(),
            com.google.protos.sight.x.proto.Sight.Object.newBuilder()
                .setLocation("0000000000:0000000002:0000000000")
                .setIndex(5)
                .addAncestorStartLocation("0000000000")
                .addAncestorStartLocation("0000000000:0000000002")
                .addAncestorStartLocation("0000000000:0000000002:0000000000")
                .setFile("SightProtoTest.java")
                .setLine(baseEventLoc.getLineNumber() + 13)
                .setFunc("logMesgWithStrFieldToCapacitorFile")
                .addAllAttribute(blockAttrs)
                .setSubType(SubType.ST_VALUE)
                .setValue(Value.newBuilder().setSubType(Value.SubType.ST_INT64).setInt64Value(123))
                .setOrder(Order.newBuilder().setTimestampNs(1000006))
                .build(),
            com.google.protos.sight.x.proto.Sight.Object.newBuilder()
                .setLocation("0000000000:0000000003")
                .setIndex(6)
                .addAncestorStartLocation("0000000000")
                .addAncestorStartLocation("0000000000:0000000003")
                .setFile("SightProtoTest.java")
                .setLine(baseEventLoc.getLineNumber() + 13)
                .setFunc("logMesgWithStrFieldToCapacitorFile")
                .addAllAttribute(blockAttrs)
                .setSubType(SubType.ST_BLOCK_END)
                .setBlockEnd(
                    BlockEnd.newBuilder()
                        .setLabel("int_val")
                        .setLocationOfBlockStart("0000000000:0000000002")
                        .setNumDirectContents(1)
                        .setNumTransitiveContents(1)
                        .setSubType(BlockEnd.SubType.ST_NAMED_VALUE)
                        .setMetrics(BlockEnd.Metrics.newBuilder().setElapsedTimeNs(2)))
                .setOrder(Order.newBuilder().setTimestampNs(1000007))
                .build(),
            com.google.protos.sight.x.proto.Sight.Object.newBuilder()
                .setLocation("0000000001")
                .setIndex(7)
                .addAncestorStartLocation("0000000001")
                .setFile("SightProtoTest.java")
                .setLine(baseEventLoc.getLineNumber() + 13)
                .setFunc("logMesgWithStrFieldToCapacitorFile")
                .addAllAttribute(blockAttrs)
                .setSubType(SubType.ST_BLOCK_END)
                .setBlockEnd(
                    BlockEnd.newBuilder()
                        .setLabel("TestMessage")
                        .setLocationOfBlockStart("0000000000")
                        .setNumDirectContents(2)
                        .setNumTransitiveContents(6)
                        .setSubType(BlockEnd.SubType.ST_LIST)
                        .setMetrics(BlockEnd.Metrics.newBuilder().setElapsedTimeNs(7)))
                .setOrder(Order.newBuilder().setTimestampNs(1000008))
                .build());
  }

  @Test
  public void logRepeatedSubMesgToTextFile() throws CapacitorException, IOException {
    // SET-UP
    Path testDir =
        Files.createDirectories(
            Path.of(tempFolder.newFolder("logRepeatedSubMesgToTextFile").getAbsolutePath()));
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
      Proto.of(
              TestMessage.newBuilder()
                  .addSubMessage(
                      TestMessage.SubMessage.newBuilder().addDoubleVal(1.23).addDoubleVal(2.34))
                  .addSubMessage(TestMessage.SubMessage.newBuilder().addDoubleVal(4.56))
                  .build())
          .log(sight);
    }

    // ASSERT
    String attrValues =
        "| class=com.google.googlex.cortex.sight.SightProtoTest,function=logRepeatedSubMesgToTextFile,"
            + "runStartTime=1969-12-31T16:00:00.001,taskBNS=";
    assertThat(FileUtils.readFileToString(new File(logFilePath), UTF_8))
        .isEqualTo(
            String.format(
                "TestMessage<<<%s\n"
                    + "TestMessage: repeated<<<%s\n"
                    + "TestMessage: repeated: SubMessage<<<%s\n"
                    + "TestMessage: repeated: SubMessage: repeated<<<%s\n"
                    + "1.230000\n"
                    + "2.340000\n"
                    + "TestMessage: repeated: SubMessage: repeated>>>>%s\n"
                    + "TestMessage: repeated: SubMessage>>>>%s\n"
                    + "TestMessage: repeated: SubMessage<<<%s\n"
                    + "TestMessage: repeated: SubMessage: repeated<<<%s\n"
                    + "4.560000\n"
                    + "TestMessage: repeated: SubMessage: repeated>>>>%s\n"
                    + "TestMessage: repeated: SubMessage>>>>%s\n"
                    + "TestMessage: repeated>>>>%s\n"
                    + "TestMessage>>>>%s\n",
                attrValues,
                attrValues,
                attrValues,
                attrValues,
                attrValues,
                attrValues,
                attrValues,
                attrValues,
                attrValues,
                attrValues,
                attrValues,
                attrValues));
  }

  @Test
  public void logRepeatedSubMesgToCapacitorFile() throws CapacitorException, IOException {
    // SET-UP
    Path testDir =
        Files.createDirectories(
            Path.of(tempFolder.newFolder("logRepeatedSubMesgToCapacitorFile").getAbsolutePath()));
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
      Proto.of(
              TestMessage.newBuilder()
                  .addSubMessage(
                      TestMessage.SubMessage.newBuilder().addDoubleVal(1.23).addDoubleVal(2.34))
                  .addSubMessage(TestMessage.SubMessage.newBuilder().addDoubleVal(4.56))
                  .build())
          .log(sight);
    }

    // ASSERT
    ImmutableList<com.google.protos.sight.x.proto.Sight.Attribute> blockAttrs =
        ImmutableList.of(
            com.google.protos.sight.x.proto.Sight.Attribute.newBuilder()
                .setKey("class")
                .setValue("com.google.googlex.cortex.sight.SightProtoTest")
                .build(),
            com.google.protos.sight.x.proto.Sight.Attribute.newBuilder()
                .setKey("function")
                .setValue("logRepeatedSubMesgToCapacitorFile")
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
                .setFile("SightProtoTest.java")
                .setLine(baseEventLoc.getLineNumber() + 19)
                .setFunc("logRepeatedSubMesgToCapacitorFile")
                .addAllAttribute(blockAttrs)
                .setSubType(SubType.ST_BLOCK_START)
                .setBlockStart(
                    BlockStart.newBuilder()
                        .setLabel("TestMessage")
                        .setSubType(BlockStart.SubType.ST_LIST)
                        .setList(ListStart.newBuilder().setSubType(ListStart.SubType.ST_DICT)))
                .setOrder(Order.newBuilder().setTimestampNs(1000001))
                .build(),
            com.google.protos.sight.x.proto.Sight.Object.newBuilder()
                .setLocation("0000000000:0000000000")
                .setIndex(1)
                .addAncestorStartLocation("0000000000")
                .addAncestorStartLocation("0000000000:0000000000")
                .setFile("SightProtoTest.java")
                .setLine(baseEventLoc.getLineNumber() + 19)
                .setFunc("logRepeatedSubMesgToCapacitorFile")
                .addAllAttribute(blockAttrs)
                .setSubType(SubType.ST_BLOCK_START)
                .setBlockStart(
                    BlockStart.newBuilder()
                        .setLabel("repeated")
                        .setSubType(BlockStart.SubType.ST_LIST)
                        .setList(
                            ListStart.newBuilder().setSubType(ListStart.SubType.ST_HOMOGENEOUS)))
                .setOrder(Order.newBuilder().setTimestampNs(1000002))
                .build(),
            com.google.protos.sight.x.proto.Sight.Object.newBuilder()
                .setLocation("0000000000:0000000000:0000000000")
                .setIndex(2)
                .addAncestorStartLocation("0000000000")
                .addAncestorStartLocation("0000000000:0000000000")
                .addAncestorStartLocation("0000000000:0000000000:0000000000")
                .setFile("SightProtoTest.java")
                .setLine(baseEventLoc.getLineNumber() + 19)
                .setFunc("logRepeatedSubMesgToCapacitorFile")
                .addAllAttribute(blockAttrs)
                .setSubType(SubType.ST_BLOCK_START)
                .setBlockStart(
                    BlockStart.newBuilder()
                        .setLabel("SubMessage")
                        .setSubType(BlockStart.SubType.ST_LIST)
                        .setList(ListStart.newBuilder().setSubType(ListStart.SubType.ST_DICT)))
                .setOrder(Order.newBuilder().setTimestampNs(1000003))
                .build(),
            com.google.protos.sight.x.proto.Sight.Object.newBuilder()
                .setLocation("0000000000:0000000000:0000000000:0000000000")
                .setIndex(3)
                .addAncestorStartLocation("0000000000")
                .addAncestorStartLocation("0000000000:0000000000")
                .addAncestorStartLocation("0000000000:0000000000:0000000000")
                .addAncestorStartLocation("0000000000:0000000000:0000000000:0000000000")
                .setFile("SightProtoTest.java")
                .setLine(baseEventLoc.getLineNumber() + 19)
                .setFunc("logRepeatedSubMesgToCapacitorFile")
                .addAllAttribute(blockAttrs)
                .setSubType(SubType.ST_BLOCK_START)
                .setBlockStart(
                    BlockStart.newBuilder()
                        .setLabel("repeated")
                        .setSubType(BlockStart.SubType.ST_LIST)
                        .setList(
                            ListStart.newBuilder().setSubType(ListStart.SubType.ST_HOMOGENEOUS)))
                .setOrder(Order.newBuilder().setTimestampNs(1000004))
                .build(),
            com.google.protos.sight.x.proto.Sight.Object.newBuilder()
                .setLocation("0000000000:0000000000:0000000000:0000000000:0000000000")
                .setIndex(4)
                .addAncestorStartLocation("0000000000")
                .addAncestorStartLocation("0000000000:0000000000")
                .addAncestorStartLocation("0000000000:0000000000:0000000000")
                .addAncestorStartLocation("0000000000:0000000000:0000000000:0000000000")
                .addAncestorStartLocation("0000000000:0000000000:0000000000:0000000000:0000000000")
                .setFile("SightProtoTest.java")
                .setLine(baseEventLoc.getLineNumber() + 19)
                .setFunc("logRepeatedSubMesgToCapacitorFile")
                .addAllAttribute(blockAttrs)
                .setSubType(SubType.ST_VALUE)
                .setValue(
                    Value.newBuilder().setSubType(Value.SubType.ST_DOUBLE).setDoubleValue(1.23))
                .setOrder(Order.newBuilder().setTimestampNs(1000005))
                .build(),
            com.google.protos.sight.x.proto.Sight.Object.newBuilder()
                .setLocation("0000000000:0000000000:0000000000:0000000000:0000000001")
                .setIndex(5)
                .addAncestorStartLocation("0000000000")
                .addAncestorStartLocation("0000000000:0000000000")
                .addAncestorStartLocation("0000000000:0000000000:0000000000")
                .addAncestorStartLocation("0000000000:0000000000:0000000000:0000000000")
                .addAncestorStartLocation("0000000000:0000000000:0000000000:0000000000:0000000001")
                .setFile("SightProtoTest.java")
                .setLine(baseEventLoc.getLineNumber() + 19)
                .setFunc("logRepeatedSubMesgToCapacitorFile")
                .addAllAttribute(blockAttrs)
                .setSubType(SubType.ST_VALUE)
                .setValue(
                    Value.newBuilder().setSubType(Value.SubType.ST_DOUBLE).setDoubleValue(2.34))
                .setOrder(Order.newBuilder().setTimestampNs(1000006))
                .build(),
            com.google.protos.sight.x.proto.Sight.Object.newBuilder()
                .setLocation("0000000000:0000000000:0000000000:0000000001")
                .setIndex(6)
                .addAncestorStartLocation("0000000000")
                .addAncestorStartLocation("0000000000:0000000000")
                .addAncestorStartLocation("0000000000:0000000000:0000000000")
                .addAncestorStartLocation("0000000000:0000000000:0000000000:0000000001")
                .setFile("SightProtoTest.java")
                .setLine(baseEventLoc.getLineNumber() + 19)
                .setFunc("logRepeatedSubMesgToCapacitorFile")
                .addAllAttribute(blockAttrs)
                .setSubType(SubType.ST_BLOCK_END)
                .setBlockEnd(
                    BlockEnd.newBuilder()
                        .setLabel("repeated")
                        .setLocationOfBlockStart("0000000000:0000000000:0000000000:0000000000")
                        .setNumDirectContents(2)
                        .setNumTransitiveContents(2)
                        .setSubType(BlockEnd.SubType.ST_LIST)
                        .setMetrics(BlockEnd.Metrics.newBuilder().setElapsedTimeNs(3)))
                .setOrder(Order.newBuilder().setTimestampNs(1000007))
                .build(),
            com.google.protos.sight.x.proto.Sight.Object.newBuilder()
                .setLocation("0000000000:0000000000:0000000001")
                .setIndex(7)
                .addAncestorStartLocation("0000000000")
                .addAncestorStartLocation("0000000000:0000000000")
                .addAncestorStartLocation("0000000000:0000000000:0000000001")
                .setFile("SightProtoTest.java")
                .setLine(baseEventLoc.getLineNumber() + 19)
                .setFunc("logRepeatedSubMesgToCapacitorFile")
                .addAllAttribute(blockAttrs)
                .setSubType(SubType.ST_BLOCK_END)
                .setBlockEnd(
                    BlockEnd.newBuilder()
                        .setLabel("SubMessage")
                        .setLocationOfBlockStart("0000000000:0000000000:0000000000")
                        .setNumDirectContents(1)
                        .setNumTransitiveContents(4)
                        .setSubType(BlockEnd.SubType.ST_LIST)
                        .setMetrics(BlockEnd.Metrics.newBuilder().setElapsedTimeNs(5)))
                .setOrder(Order.newBuilder().setTimestampNs(1000008))
                .build(),
            com.google.protos.sight.x.proto.Sight.Object.newBuilder()
                .setLocation("0000000000:0000000000:0000000002")
                .setIndex(8)
                .addAncestorStartLocation("0000000000")
                .addAncestorStartLocation("0000000000:0000000000")
                .addAncestorStartLocation("0000000000:0000000000:0000000002")
                .setFile("SightProtoTest.java")
                .setLine(baseEventLoc.getLineNumber() + 19)
                .setFunc("logRepeatedSubMesgToCapacitorFile")
                .addAllAttribute(blockAttrs)
                .setSubType(SubType.ST_BLOCK_START)
                .setBlockStart(
                    BlockStart.newBuilder()
                        .setLabel("SubMessage")
                        .setSubType(BlockStart.SubType.ST_LIST)
                        .setList(ListStart.newBuilder().setSubType(ListStart.SubType.ST_DICT)))
                .setOrder(Order.newBuilder().setTimestampNs(1000009))
                .build(),
            com.google.protos.sight.x.proto.Sight.Object.newBuilder()
                .setLocation("0000000000:0000000000:0000000002:0000000000")
                .setIndex(9)
                .addAncestorStartLocation("0000000000")
                .addAncestorStartLocation("0000000000:0000000000")
                .addAncestorStartLocation("0000000000:0000000000:0000000002")
                .addAncestorStartLocation("0000000000:0000000000:0000000002:0000000000")
                .setFile("SightProtoTest.java")
                .setLine(baseEventLoc.getLineNumber() + 19)
                .setFunc("logRepeatedSubMesgToCapacitorFile")
                .addAllAttribute(blockAttrs)
                .setSubType(SubType.ST_BLOCK_START)
                .setBlockStart(
                    BlockStart.newBuilder()
                        .setLabel("repeated")
                        .setSubType(BlockStart.SubType.ST_LIST)
                        .setList(
                            ListStart.newBuilder().setSubType(ListStart.SubType.ST_HOMOGENEOUS)))
                .setOrder(Order.newBuilder().setTimestampNs(1000010))
                .build(),
            com.google.protos.sight.x.proto.Sight.Object.newBuilder()
                .setLocation("0000000000:0000000000:0000000002:0000000000:0000000000")
                .setIndex(10)
                .addAncestorStartLocation("0000000000")
                .addAncestorStartLocation("0000000000:0000000000")
                .addAncestorStartLocation("0000000000:0000000000:0000000002")
                .addAncestorStartLocation("0000000000:0000000000:0000000002:0000000000")
                .addAncestorStartLocation("0000000000:0000000000:0000000002:0000000000:0000000000")
                .setFile("SightProtoTest.java")
                .setLine(baseEventLoc.getLineNumber() + 19)
                .setFunc("logRepeatedSubMesgToCapacitorFile")
                .addAllAttribute(blockAttrs)
                .setSubType(SubType.ST_VALUE)
                .setValue(
                    Value.newBuilder().setSubType(Value.SubType.ST_DOUBLE).setDoubleValue(4.56))
                .setOrder(Order.newBuilder().setTimestampNs(1000011))
                .build(),
            com.google.protos.sight.x.proto.Sight.Object.newBuilder()
                .setLocation("0000000000:0000000000:0000000002:0000000001")
                .setIndex(11)
                .addAncestorStartLocation("0000000000")
                .addAncestorStartLocation("0000000000:0000000000")
                .addAncestorStartLocation("0000000000:0000000000:0000000002")
                .addAncestorStartLocation("0000000000:0000000000:0000000002:0000000001")
                .setFile("SightProtoTest.java")
                .setLine(baseEventLoc.getLineNumber() + 19)
                .setFunc("logRepeatedSubMesgToCapacitorFile")
                .addAllAttribute(blockAttrs)
                .setSubType(SubType.ST_BLOCK_END)
                .setBlockEnd(
                    BlockEnd.newBuilder()
                        .setLabel("repeated")
                        .setLocationOfBlockStart("0000000000:0000000000:0000000002:0000000000")
                        .setNumDirectContents(1)
                        .setNumTransitiveContents(1)
                        .setSubType(BlockEnd.SubType.ST_LIST)
                        .setMetrics(BlockEnd.Metrics.newBuilder().setElapsedTimeNs(2)))
                .setOrder(Order.newBuilder().setTimestampNs(1000012))
                .build(),
            com.google.protos.sight.x.proto.Sight.Object.newBuilder()
                .setLocation("0000000000:0000000000:0000000003")
                .setIndex(12)
                .addAncestorStartLocation("0000000000")
                .addAncestorStartLocation("0000000000:0000000000")
                .addAncestorStartLocation("0000000000:0000000000:0000000003")
                .setFile("SightProtoTest.java")
                .setLine(baseEventLoc.getLineNumber() + 19)
                .setFunc("logRepeatedSubMesgToCapacitorFile")
                .addAllAttribute(blockAttrs)
                .setSubType(SubType.ST_BLOCK_END)
                .setBlockEnd(
                    BlockEnd.newBuilder()
                        .setLabel("SubMessage")
                        .setLocationOfBlockStart("0000000000:0000000000:0000000002")
                        .setNumDirectContents(1)
                        .setNumTransitiveContents(3)
                        .setSubType(BlockEnd.SubType.ST_LIST)
                        .setMetrics(BlockEnd.Metrics.newBuilder().setElapsedTimeNs(4)))
                .setOrder(Order.newBuilder().setTimestampNs(1000013))
                .build(),
            com.google.protos.sight.x.proto.Sight.Object.newBuilder()
                .setLocation("0000000000:0000000001")
                .setIndex(13)
                .addAncestorStartLocation("0000000000")
                .addAncestorStartLocation("0000000000:0000000001")
                .setFile("SightProtoTest.java")
                .setLine(baseEventLoc.getLineNumber() + 19)
                .setFunc("logRepeatedSubMesgToCapacitorFile")
                .addAllAttribute(blockAttrs)
                .setSubType(SubType.ST_BLOCK_END)
                .setBlockEnd(
                    BlockEnd.newBuilder()
                        .setLabel("repeated")
                        .setLocationOfBlockStart("0000000000:0000000000")
                        .setNumDirectContents(2)
                        .setNumTransitiveContents(11)
                        .setSubType(BlockEnd.SubType.ST_LIST)
                        .setMetrics(BlockEnd.Metrics.newBuilder().setElapsedTimeNs(12)))
                .setOrder(Order.newBuilder().setTimestampNs(1000014))
                .build(),
            com.google.protos.sight.x.proto.Sight.Object.newBuilder()
                .setLocation("0000000001")
                .setIndex(14)
                .addAncestorStartLocation("0000000001")
                .setFile("SightProtoTest.java")
                .setLine(baseEventLoc.getLineNumber() + 19)
                .setFunc("logRepeatedSubMesgToCapacitorFile")
                .addAllAttribute(blockAttrs)
                .setSubType(SubType.ST_BLOCK_END)
                .setBlockEnd(
                    BlockEnd.newBuilder()
                        .setLabel("TestMessage")
                        .setLocationOfBlockStart("0000000000")
                        .setNumDirectContents(1)
                        .setNumTransitiveContents(13)
                        .setSubType(BlockEnd.SubType.ST_LIST)
                        .setMetrics(BlockEnd.Metrics.newBuilder().setElapsedTimeNs(14)))
                .setOrder(Order.newBuilder().setTimestampNs(1000015))
                .build());
  }

  @Test
  public void logListOfMesgsWithStrFieldToTextFile() throws CapacitorException, IOException {
    // SET-UP
    Path testDir =
        Files.createDirectories(
            Path.of(
                tempFolder.newFolder("logListOfMesgsWithStrFieldToTextFile").getAbsolutePath()));
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
      Collections.log(
          ImmutableList.of(
              Proto.of(TestMessage.newBuilder().setStrVal("a").build()),
              Proto.of(TestMessage.newBuilder().setStrVal("b").build())),
          sight);
    }

    // ASSERT
    String attrValues =
        "| class=com.google.googlex.cortex.sight.SightProtoTest,function=logListOfMesgsWithStrFieldToTextFile,"
            + "runStartTime=1969-12-31T16:00:00.001,taskBNS=";
    assertThat(FileUtils.readFileToString(new File(logFilePath), UTF_8))
        .isEqualTo(
            String.format(
                "list<<<%s\n"
                    + "list: TestMessage<<<%s\n"
                    + "list: TestMessage: str_val<<<%s\n"
                    + "a\n"
                    + "list: TestMessage: str_val>>>>%s\n"
                    + "list: TestMessage>>>>%s\n"
                    + "list: TestMessage<<<%s\n"
                    + "list: TestMessage: str_val<<<%s\n"
                    + "b\n"
                    + "list: TestMessage: str_val>>>>%s\n"
                    + "list: TestMessage>>>>%s\n"
                    + "list>>>>%s\n",
                attrValues,
                attrValues,
                attrValues,
                attrValues,
                attrValues,
                attrValues,
                attrValues,
                attrValues,
                attrValues,
                attrValues));
  }

  @Test
  public void logListOfMesgsWithStrFieldToCapacitorFile() throws CapacitorException, IOException {
    // SET-UP
    Path testDir =
        Files.createDirectories(
            Path.of(
                tempFolder
                    .newFolder("logListOfMesgsWithStrFieldToCapacitorFile")
                    .getAbsolutePath()));
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
          ImmutableList.of(
              Proto.of(TestMessage.newBuilder().setStrVal("a").build()),
              Proto.of(TestMessage.newBuilder().setStrVal("b").build())),
          sight);
    }

    // ASSERT
    ImmutableList<com.google.protos.sight.x.proto.Sight.Attribute> blockAttrs =
        ImmutableList.of(
            com.google.protos.sight.x.proto.Sight.Attribute.newBuilder()
                .setKey("class")
                .setValue("com.google.googlex.cortex.sight.SightProtoTest")
                .build(),
            com.google.protos.sight.x.proto.Sight.Attribute.newBuilder()
                .setKey("function")
                .setValue("logListOfMesgsWithStrFieldToCapacitorFile")
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
                .setFile("SightProtoTest.java")
                .setLine(baseEventLoc.getLineNumber() + 13)
                .setFunc("logListOfMesgsWithStrFieldToCapacitorFile")
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
                .setFile("SightProtoTest.java")
                .setLine(baseEventLoc.getLineNumber() + 13)
                .setFunc("logListOfMesgsWithStrFieldToCapacitorFile")
                .addAllAttribute(blockAttrs)
                .setSubType(SubType.ST_BLOCK_START)
                .setBlockStart(
                    BlockStart.newBuilder()
                        .setLabel("TestMessage")
                        .setSubType(BlockStart.SubType.ST_LIST)
                        .setList(ListStart.newBuilder().setSubType(ListStart.SubType.ST_DICT)))
                .setOrder(Order.newBuilder().setTimestampNs(1000002))
                .build(),
            com.google.protos.sight.x.proto.Sight.Object.newBuilder()
                .setLocation("0000000000:0000000000:0000000000")
                .setIndex(2)
                .addAncestorStartLocation("0000000000")
                .addAncestorStartLocation("0000000000:0000000000")
                .addAncestorStartLocation("0000000000:0000000000:0000000000")
                .setFile("SightProtoTest.java")
                .setLine(baseEventLoc.getLineNumber() + 13)
                .setFunc("logListOfMesgsWithStrFieldToCapacitorFile")
                .addAllAttribute(blockAttrs)
                .setSubType(SubType.ST_BLOCK_START)
                .setBlockStart(
                    BlockStart.newBuilder()
                        .setLabel("str_val")
                        .setSubType(BlockStart.SubType.ST_NAMED_VALUE))
                .setOrder(Order.newBuilder().setTimestampNs(1000003))
                .build(),
            com.google.protos.sight.x.proto.Sight.Object.newBuilder()
                .setLocation("0000000000:0000000000:0000000000:0000000000")
                .setIndex(3)
                .addAncestorStartLocation("0000000000")
                .addAncestorStartLocation("0000000000:0000000000")
                .addAncestorStartLocation("0000000000:0000000000:0000000000")
                .addAncestorStartLocation("0000000000:0000000000:0000000000:0000000000")
                .setFile("SightProtoTest.java")
                .setLine(baseEventLoc.getLineNumber() + 13)
                .setFunc("logListOfMesgsWithStrFieldToCapacitorFile")
                .addAllAttribute(blockAttrs)
                .setSubType(SubType.ST_VALUE)
                .setValue(
                    Value.newBuilder().setSubType(Value.SubType.ST_STRING).setStringValue("a"))
                .setOrder(Order.newBuilder().setTimestampNs(1000004))
                .build(),
            com.google.protos.sight.x.proto.Sight.Object.newBuilder()
                .setLocation("0000000000:0000000000:0000000001")
                .setIndex(4)
                .addAncestorStartLocation("0000000000")
                .addAncestorStartLocation("0000000000:0000000000")
                .addAncestorStartLocation("0000000000:0000000000:0000000001")
                .setFile("SightProtoTest.java")
                .setLine(baseEventLoc.getLineNumber() + 13)
                .setFunc("logListOfMesgsWithStrFieldToCapacitorFile")
                .addAllAttribute(blockAttrs)
                .setSubType(SubType.ST_BLOCK_END)
                .setBlockEnd(
                    BlockEnd.newBuilder()
                        .setLabel("str_val")
                        .setLocationOfBlockStart("0000000000:0000000000:0000000000")
                        .setNumDirectContents(1)
                        .setNumTransitiveContents(1)
                        .setSubType(BlockEnd.SubType.ST_NAMED_VALUE)
                        .setMetrics(BlockEnd.Metrics.newBuilder().setElapsedTimeNs(2)))
                .setOrder(Order.newBuilder().setTimestampNs(1000005))
                .build(),
            com.google.protos.sight.x.proto.Sight.Object.newBuilder()
                .setLocation("0000000000:0000000001")
                .setIndex(5)
                .addAncestorStartLocation("0000000000")
                .addAncestorStartLocation("0000000000:0000000001")
                .setFile("SightProtoTest.java")
                .setLine(baseEventLoc.getLineNumber() + 13)
                .setFunc("logListOfMesgsWithStrFieldToCapacitorFile")
                .addAllAttribute(blockAttrs)
                .setSubType(SubType.ST_BLOCK_END)
                .setBlockEnd(
                    BlockEnd.newBuilder()
                        .setLabel("TestMessage")
                        .setLocationOfBlockStart("0000000000:0000000000")
                        .setNumDirectContents(1)
                        .setNumTransitiveContents(3)
                        .setSubType(BlockEnd.SubType.ST_LIST)
                        .setMetrics(BlockEnd.Metrics.newBuilder().setElapsedTimeNs(4)))
                .setOrder(Order.newBuilder().setTimestampNs(1000006))
                .build(),
            com.google.protos.sight.x.proto.Sight.Object.newBuilder()
                .setLocation("0000000000:0000000002")
                .setIndex(6)
                .addAncestorStartLocation("0000000000")
                .addAncestorStartLocation("0000000000:0000000002")
                .setFile("SightProtoTest.java")
                .setLine(baseEventLoc.getLineNumber() + 13)
                .setFunc("logListOfMesgsWithStrFieldToCapacitorFile")
                .addAllAttribute(blockAttrs)
                .setSubType(SubType.ST_BLOCK_START)
                .setBlockStart(
                    BlockStart.newBuilder()
                        .setLabel("TestMessage")
                        .setSubType(BlockStart.SubType.ST_LIST)
                        .setList(ListStart.newBuilder().setSubType(ListStart.SubType.ST_DICT)))
                .setOrder(Order.newBuilder().setTimestampNs(1000007))
                .build(),
            com.google.protos.sight.x.proto.Sight.Object.newBuilder()
                .setLocation("0000000000:0000000002:0000000000")
                .setIndex(7)
                .addAncestorStartLocation("0000000000")
                .addAncestorStartLocation("0000000000:0000000002")
                .addAncestorStartLocation("0000000000:0000000002:0000000000")
                .setFile("SightProtoTest.java")
                .setLine(baseEventLoc.getLineNumber() + 13)
                .setFunc("logListOfMesgsWithStrFieldToCapacitorFile")
                .addAllAttribute(blockAttrs)
                .setSubType(SubType.ST_BLOCK_START)
                .setBlockStart(
                    BlockStart.newBuilder()
                        .setLabel("str_val")
                        .setSubType(BlockStart.SubType.ST_NAMED_VALUE))
                .setOrder(Order.newBuilder().setTimestampNs(1000008))
                .build(),
            com.google.protos.sight.x.proto.Sight.Object.newBuilder()
                .setLocation("0000000000:0000000002:0000000000:0000000000")
                .setIndex(8)
                .addAncestorStartLocation("0000000000")
                .addAncestorStartLocation("0000000000:0000000002")
                .addAncestorStartLocation("0000000000:0000000002:0000000000")
                .addAncestorStartLocation("0000000000:0000000002:0000000000:0000000000")
                .setFile("SightProtoTest.java")
                .setLine(baseEventLoc.getLineNumber() + 13)
                .setFunc("logListOfMesgsWithStrFieldToCapacitorFile")
                .addAllAttribute(blockAttrs)
                .setSubType(SubType.ST_VALUE)
                .setValue(
                    Value.newBuilder().setSubType(Value.SubType.ST_STRING).setStringValue("b"))
                .setOrder(Order.newBuilder().setTimestampNs(1000009))
                .build(),
            com.google.protos.sight.x.proto.Sight.Object.newBuilder()
                .setLocation("0000000000:0000000002:0000000001")
                .setIndex(9)
                .addAncestorStartLocation("0000000000")
                .addAncestorStartLocation("0000000000:0000000002")
                .addAncestorStartLocation("0000000000:0000000002:0000000001")
                .setFile("SightProtoTest.java")
                .setLine(baseEventLoc.getLineNumber() + 13)
                .setFunc("logListOfMesgsWithStrFieldToCapacitorFile")
                .addAllAttribute(blockAttrs)
                .setSubType(SubType.ST_BLOCK_END)
                .setBlockEnd(
                    BlockEnd.newBuilder()
                        .setLabel("str_val")
                        .setLocationOfBlockStart("0000000000:0000000002:0000000000")
                        .setNumDirectContents(1)
                        .setNumTransitiveContents(1)
                        .setSubType(BlockEnd.SubType.ST_NAMED_VALUE)
                        .setMetrics(BlockEnd.Metrics.newBuilder().setElapsedTimeNs(2)))
                .setOrder(Order.newBuilder().setTimestampNs(1000010))
                .build(),
            com.google.protos.sight.x.proto.Sight.Object.newBuilder()
                .setLocation("0000000000:0000000003")
                .setIndex(10)
                .addAncestorStartLocation("0000000000")
                .addAncestorStartLocation("0000000000:0000000003")
                .setFile("SightProtoTest.java")
                .setLine(baseEventLoc.getLineNumber() + 13)
                .setFunc("logListOfMesgsWithStrFieldToCapacitorFile")
                .addAllAttribute(blockAttrs)
                .setSubType(SubType.ST_BLOCK_END)
                .setBlockEnd(
                    BlockEnd.newBuilder()
                        .setLabel("TestMessage")
                        .setLocationOfBlockStart("0000000000:0000000002")
                        .setNumDirectContents(1)
                        .setNumTransitiveContents(3)
                        .setSubType(BlockEnd.SubType.ST_LIST)
                        .setMetrics(BlockEnd.Metrics.newBuilder().setElapsedTimeNs(4)))
                .setOrder(Order.newBuilder().setTimestampNs(1000011))
                .build(),
            com.google.protos.sight.x.proto.Sight.Object.newBuilder()
                .setLocation("0000000001")
                .setIndex(11)
                .addAncestorStartLocation("0000000001")
                .setFile("SightProtoTest.java")
                .setLine(baseEventLoc.getLineNumber() + 13)
                .setFunc("logListOfMesgsWithStrFieldToCapacitorFile")
                .addAllAttribute(blockAttrs)
                .setSubType(SubType.ST_BLOCK_END)
                .setBlockEnd(
                    BlockEnd.newBuilder()
                        .setLabel("list")
                        .setLocationOfBlockStart("0000000000")
                        .setNumDirectContents(2)
                        .setNumTransitiveContents(10)
                        .setSubType(BlockEnd.SubType.ST_LIST)
                        .setMetrics(BlockEnd.Metrics.newBuilder().setElapsedTimeNs(11)))
                .setOrder(Order.newBuilder().setTimestampNs(1000012))
                .build());
  }
}
