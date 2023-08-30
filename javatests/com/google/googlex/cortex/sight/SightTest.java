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
import com.google.googlex.cortex.sight.sightservice.SightServiceGrpc.SightServiceImplBase;
import com.google.googlex.cortex.sight.sightservice.SightServiceOuterClass.CreateRequest;
import com.google.googlex.cortex.sight.sightservice.SightServiceOuterClass.CreateResponse;
import com.google.googlex.cortex.sight.sightservice.SightServiceOuterClass.FinalizeRequest;
import com.google.googlex.cortex.sight.sightservice.SightServiceOuterClass.FinalizeResponse;
import com.google.protobuf.Descriptors.FieldDescriptor;
import com.google.protobuf.ExtensionRegistry;
import com.google.protos.sight.x.proto.Sight.BlockEnd;
import com.google.protos.sight.x.proto.Sight.BlockStart;
import com.google.protos.sight.x.proto.Sight.Object.Metrics;
import com.google.protos.sight.x.proto.Sight.Object.Order;
import com.google.protos.sight.x.proto.Sight.Object.SubType;
import com.google.protos.sight.x.proto.Sight.Params;
import com.google.protos.sight.x.proto.Sight.Text;
import com.google.testing.util.AssertingHandler;
import io.grpc.Context;
import io.grpc.Deadline;
import io.grpc.stub.StreamObserver;
import io.grpc.testing.GrpcServerRule;
import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.time.Duration;
import java.time.Instant;
import java.util.ArrayList;
import java.util.List;
import java.util.Optional;
import java.util.concurrent.atomic.AtomicReference;
import java.util.logging.Logger;
import org.apache.commons.io.FileUtils;
import org.junit.After;
import org.junit.Before;
import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.TemporaryFolder;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

@RunWith(JUnit4.class)
public final class SightTest {
  @Rule public final TemporaryFolder tempFolder = new TemporaryFolder();
  private static final GoogleLogger logger = GoogleLogger.forEnclosingClass();
  @Rule public final GrpcServerRule grpcServerRule = new GrpcServerRule().directExecutor();

  private final AssertingHandler assertingHandler = new AssertingHandler();

  private static final ImmutableList<FieldDescriptor> IGNORABLE_FIELDS =
      ImmutableList.of(
          com.google.protos.sight.x.proto.Sight.Object.getDescriptor()
              .findFieldByNumber(com.google.protos.sight.x.proto.Sight.Object.METRICS_FIELD_NUMBER),
          Metrics.getDescriptor()
              .findFieldByNumber(Metrics.PROCESS_FREE_SWAP_SPACE_BYTES_FIELD_NUMBER),
          Metrics.getDescriptor()
              .findFieldByNumber(Metrics.PROCESS_TOTAL_SWAP_SPACE_BYTES_FIELD_NUMBER));

  @Before
  public void setUp() {
    Logger.getLogger("com.google.codelab.grpc").addHandler(assertingHandler);
  }

  @After
  public void tearDown() {
    Logger.getLogger("com.google.codelab.grpc").removeHandler(assertingHandler);
    assertingHandler.close();
  }

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
  public void logTextToTextFile() throws CapacitorException, IOException {
    // SET-UP
    Path testDir =
        Files.createDirectories(
            Path.of(tempFolder.newFolder("logTextToTextFile").getAbsolutePath()));
    String logFilePath;
    StackTraceElement baseEventLoc = Thread.currentThread().getStackTrace()[1];
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
      sight.text("textA");
      Sight.text("textB", sight);
      sight.textLine("textLine1");
      Sight.textLine("textLine2", sight);
    }

    // ASSERT
    assertThat(FileUtils.readFileToString(new File(logFilePath), UTF_8))
        .isEqualTo(
            String.format(
                "textAtextBSightTest.java:%d/logTextToTextFile textLine1|"
                    + " runStartTime=1969-12-31T16:00:00.001,taskBNS=\n"
                    + "SightTest.java:%d/logTextToTextFile textLine2|"
                    + " runStartTime=1969-12-31T16:00:00.001,taskBNS=\n",
                baseEventLoc.getLineNumber() + 15, baseEventLoc.getLineNumber() + 16));
  }

  @Test
  public void logTextToCapacitorFile() throws CapacitorException, IOException {
    // SET-UP
    Path testDir =
        Files.createDirectories(
            Path.of(tempFolder.newFolder("logTextToCapacitorFile").getAbsolutePath()));
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
      sight.text("textA");
      Sight.text("textB", sight);
      sight.textLine("textLine1");
      Sight.textLine("textLine2", sight);
    }

    ImmutableList<com.google.protos.sight.x.proto.Sight.Attribute> attrs =
        ImmutableList.of(
            com.google.protos.sight.x.proto.Sight.Attribute.newBuilder()
                .setKey("runStartTime")
                .setValue("1969-12-31T16:00:00.001")
                .build(),
            com.google.protos.sight.x.proto.Sight.Attribute.newBuilder()
                .setKey("taskBNS")
                .setValue("")
                .build());

    // ASSERT
    assertThat(loadLog(logFilePath))
        .ignoringFieldDescriptors(IGNORABLE_FIELDS)
        .containsExactly(
            com.google.protos.sight.x.proto.Sight.Object.newBuilder()
                .setLocation("0000000000")
                .setIndex(0)
                .addAncestorStartLocation("0000000000")
                .setFile("SightTest.java")
                .setLine(baseEventLoc.getLineNumber() + 13)
                .setFunc("logTextToCapacitorFile")
                .addAllAttribute(attrs)
                .setSubType(SubType.ST_TEXT)
                .setText(Text.newBuilder().setText("textA"))
                .setOrder(Order.newBuilder().setTimestampNs(1000001))
                .build(),
            com.google.protos.sight.x.proto.Sight.Object.newBuilder()
                .setLocation("0000000001")
                .setIndex(1)
                .addAncestorStartLocation("0000000001")
                .setFile("SightTest.java")
                .setLine(baseEventLoc.getLineNumber() + 14)
                .setFunc("logTextToCapacitorFile")
                .addAllAttribute(attrs)
                .setSubType(SubType.ST_TEXT)
                .setText(Text.newBuilder().setText("textB"))
                .setOrder(Order.newBuilder().setTimestampNs(1000002))
                .build(),
            com.google.protos.sight.x.proto.Sight.Object.newBuilder()
                .setLocation("0000000002")
                .setIndex(2)
                .addAncestorStartLocation("0000000002")
                .setFile("SightTest.java")
                .setLine(baseEventLoc.getLineNumber() + 15)
                .setFunc("logTextToCapacitorFile")
                .addAllAttribute(attrs)
                .setSubType(SubType.ST_TEXT)
                .setText(Text.newBuilder().setText("textLine1\n"))
                .setOrder(Order.newBuilder().setTimestampNs(1000003))
                .build(),
            com.google.protos.sight.x.proto.Sight.Object.newBuilder()
                .setLocation("0000000003")
                .setIndex(3)
                .addAncestorStartLocation("0000000003")
                .setFile("SightTest.java")
                .setLine(baseEventLoc.getLineNumber() + 16)
                .setFunc("logTextToCapacitorFile")
                .addAllAttribute(attrs)
                .setSubType(SubType.ST_TEXT)
                .setText(Text.newBuilder().setText("textLine2\n"))
                .setOrder(Order.newBuilder().setTimestampNs(1000004))
                .build());
  }

  @Test
  public void getOutputPathViaServer() throws CapacitorException, IOException {
    AtomicReference<Deadline> savedDeadline = new AtomicReference<>();
    AtomicReference<CreateRequest> savedCreateReq = new AtomicReference<>();
    AtomicReference<FinalizeRequest> savedFinalizeReq = new AtomicReference<>();

    Path testDir =
        Files.createDirectories(
            Path.of(tempFolder.newFolder("getOutputPathViaServer").getAbsolutePath()));
    grpcServerRule
        .getServiceRegistry()
        .addService(
            new SightServiceImplBase() {
              @Override
              public void create(CreateRequest req, StreamObserver<CreateResponse> respObserver) {
                // Set info for later. A failed assertion wouldn't propagate up to test method
                savedDeadline.set(Context.current().getDeadline());
                savedCreateReq.set(req);
                CreateResponse response =
                    CreateResponse.newBuilder()
                        .setId(123)
                        .setPathPrefix(testDir + "/server_log_label")
                        .build();
                respObserver.onNext(response);
                respObserver.onCompleted();
              }

              @Override
              public void finalize(
                  FinalizeRequest req, StreamObserver<FinalizeResponse> respObserver) {
                // Set info for later. A failed assertion wouldn't propagate up to test method
                savedDeadline.set(Context.current().getDeadline());
                savedFinalizeReq.set(req);
                respObserver.onNext(FinalizeResponse.getDefaultInstance());
                respObserver.onCompleted();
              }
            });

    // SET-UP
    String logFilePath;
    StackTraceElement baseEventLoc = Thread.currentThread().getStackTrace()[1];
    try (SightImpl sight =
        SightImpl.create(
            Params.newBuilder().setCapacitorOutput(true).build(),
            /* timeSource=*/ FakeTimeSource.create()
                .setNow(Instant.ofEpochMilli(1))
                .setAutoAdvance(Duration.ofNanos(1)),
            /* channel= */ Optional.of(grpcServerRule.getChannel()))) {
      logFilePath = sight.getCapacitorLogFilePath().toString();
      // ACT
      sight.text("text");
    }

    ImmutableList<com.google.protos.sight.x.proto.Sight.Attribute> attrs =
        ImmutableList.of(
            com.google.protos.sight.x.proto.Sight.Attribute.newBuilder()
                .setKey("runStartTime")
                .setValue("1969-12-31T16:00:00.001")
                .build(),
            com.google.protos.sight.x.proto.Sight.Attribute.newBuilder()
                .setKey("taskBNS")
                .setValue("")
                .build());

    // ASSERT
    assertThat(loadLog(logFilePath))
        .ignoringFieldDescriptors(IGNORABLE_FIELDS)
        .containsExactly(
            com.google.protos.sight.x.proto.Sight.Object.newBuilder()
                .setLocation("0000000000")
                .setIndex(0)
                .addAncestorStartLocation("0000000000")
                .setFile("SightTest.java")
                .setLine(baseEventLoc.getLineNumber() + 10)
                .setFunc("getOutputPathViaServer")
                .addAllAttribute(attrs)
                .setSubType(SubType.ST_TEXT)
                .setText(Text.newBuilder().setText("text"))
                .setOrder(Order.newBuilder().setTimestampNs(1000001))
                .build());
  }

  @Test
  public void logBlockTextToTextFile() throws CapacitorException, IOException {
    // SET-UP
    Path testDir =
        Files.createDirectories(
            Path.of(tempFolder.newFolder("logBlockTextToTextFile").getAbsolutePath()));
    String logFilePath;
    StackTraceElement baseEventLoc = Thread.currentThread().getStackTrace()[1];
    try (SightImpl sight =
            SightImpl.create(
                Params.newBuilder()
                    .setLocal(true)
                    .setTextOutput(true)
                    .setLogDirPath(testDir.toString())
                    .build(),
                /* timeSource= */ FakeTimeSource.create()
                    .setNow(Instant.ofEpochMilli(1))
                    .setAutoAdvance(Duration.ofNanos(1)));
        // ACT
        Block b = Block.create("block", sight)) {
      logFilePath = sight.getTextLogFilePath().toString();
      sight.textLine("text");
    }
    // ASSERT
    String attrs =
        "| class=com.google.googlex.cortex.sight.SightTest,function=logBlockTextToTextFile,"
            + "runStartTime=1969-12-31T16:00:00.001,taskBNS=";
    assertThat(FileUtils.readFileToString(new File(logFilePath), UTF_8))
        .isEqualTo(
            String.format(
                "block<<<%s\n"
                    + "SightTest.java:%d/logBlockTextToTextFile block: text%s\n"
                    + "block>>>>%s\n",
                attrs, baseEventLoc.getLineNumber() + 14, attrs, attrs));
  }

  @Test
  public void logBlockToCapacitorFile() throws CapacitorException, IOException {
    // SET-UP
    Path testDir =
        Files.createDirectories(
            Path.of(tempFolder.newFolder("logBlockToCapacitorFile").getAbsolutePath()));
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
                    .setAutoAdvance(Duration.ofNanos(1)));
        // ACT
        Block b = Block.create("block", sight)) {
      logFilePath = sight.getCapacitorLogFilePath().toString();
      sight.textLine("text");
    }

    // ASSERT
    ImmutableList<com.google.protos.sight.x.proto.Sight.Attribute> attrs =
        ImmutableList.of(
            com.google.protos.sight.x.proto.Sight.Attribute.newBuilder()
                .setKey("class")
                .setValue("com.google.googlex.cortex.sight.SightTest")
                .build(),
            com.google.protos.sight.x.proto.Sight.Attribute.newBuilder()
                .setKey("function")
                .setValue("logBlockToCapacitorFile")
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
                .setFile("SightTest.java")
                .setLine(baseEventLoc.getLineNumber() + 12)
                .setFunc("logBlockToCapacitorFile")
                .addAllAttribute(attrs)
                .setSubType(SubType.ST_BLOCK_START)
                .setBlockStart(BlockStart.newBuilder().setLabel("block"))
                .setOrder(Order.newBuilder().setTimestampNs(1000001))
                .build(),
            com.google.protos.sight.x.proto.Sight.Object.newBuilder()
                .setLocation("0000000000:0000000000")
                .setIndex(1)
                .addAncestorStartLocation("0000000000")
                .addAncestorStartLocation("0000000000:0000000000")
                .setFile("SightTest.java")
                .setLine(baseEventLoc.getLineNumber() + 14)
                .setFunc("logBlockToCapacitorFile")
                .addAllAttribute(attrs)
                .setSubType(SubType.ST_TEXT)
                .setText(Text.newBuilder().setText("text\n"))
                .setOrder(Order.newBuilder().setTimestampNs(1000002))
                .build(),
            com.google.protos.sight.x.proto.Sight.Object.newBuilder()
                .setLocation("0000000001")
                .setIndex(2)
                .addAncestorStartLocation("0000000001")
                .setFile("SightTest.java")
                .setLine(baseEventLoc.getLineNumber() + 15)
                .setFunc("logBlockToCapacitorFile")
                .addAllAttribute(attrs)
                .setSubType(SubType.ST_BLOCK_END)
                .setBlockEnd(
                    BlockEnd.newBuilder()
                        .setLabel("block")
                        .setLocationOfBlockStart("0000000000")
                        .setNumDirectContents(1)
                        .setNumTransitiveContents(1)
                        .setMetrics(BlockEnd.Metrics.newBuilder().setElapsedTimeNs(2)))
                .setOrder(Order.newBuilder().setTimestampNs(1000003))
                .build());
  }

  @Test
  public void logNestedBlockTextToTextFile() throws CapacitorException, IOException {
    // SET-UP
    Path testDir =
        Files.createDirectories(
            Path.of(tempFolder.newFolder("logNestedBlockTextToTextFile").getAbsolutePath()));
    String logFilePath;
    StackTraceElement baseEventLoc = Thread.currentThread().getStackTrace()[1];
    try (SightImpl sight =
            SightImpl.create(
                Params.newBuilder()
                    .setLocal(true)
                    .setTextOutput(true)
                    .setLogDirPath(testDir.toString())
                    .build(),
                /* timeSource= */ FakeTimeSource.create()
                    .setNow(Instant.ofEpochMilli(1))
                    .setAutoAdvance(Duration.ofNanos(1)));
        // ACT
        Block b1 = Block.create("outerBlock", sight)) {
      logFilePath = sight.getTextLogFilePath().toString();
      sight.textLine("preText");
      try (Block b2 = Block.create("innerBlock", sight)) {
        sight.textLine("inText");
      }
      sight.textLine("postText");
    }
    // ASSERT
    String attrs =
        "| class=com.google.googlex.cortex.sight.SightTest,function=logNestedBlockTextToTextFile,"
            + "runStartTime=1969-12-31T16:00:00.001,taskBNS=";
    assertThat(FileUtils.readFileToString(new File(logFilePath), UTF_8))
        .isEqualTo(
            String.format(
                "outerBlock<<<%s\n"
                    + "SightTest.java:%d/logNestedBlockTextToTextFile outerBlock: preText%s\n"
                    + "outerBlock: innerBlock<<<%s\n"
                    + "SightTest.java:%d/logNestedBlockTextToTextFile outerBlock: innerBlock:"
                    + " inText%s\n"
                    + "outerBlock: innerBlock>>>>%s\n"
                    + "SightTest.java:%d/logNestedBlockTextToTextFile outerBlock: postText%s\n"
                    + "outerBlock>>>>%s\n",
                attrs,
                baseEventLoc.getLineNumber() + 14,
                attrs,
                attrs,
                baseEventLoc.getLineNumber() + 16,
                attrs,
                attrs,
                baseEventLoc.getLineNumber() + 18,
                attrs,
                attrs));
  }

  @Test
  public void logNestedBlockToCapacitorFile() throws CapacitorException, IOException {
    // SET-UP
    Path testDir =
        Files.createDirectories(
            Path.of(tempFolder.newFolder("logNestedBlockToCapacitorFile").getAbsolutePath()));
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
                    .setAutoAdvance(Duration.ofNanos(1)));
        // ACT
        Block b1 = Block.create("outerBlock", sight)) {
      logFilePath = sight.getCapacitorLogFilePath().toString();
      sight.text("preText");
      try (Block b2 = Block.create("innerBlock", sight)) {
        sight.text("inText");
      }
      sight.text("postText");
    }
    // ASSERT
    ImmutableList<com.google.protos.sight.x.proto.Sight.Attribute> attrs =
        ImmutableList.of(
            com.google.protos.sight.x.proto.Sight.Attribute.newBuilder()
                .setKey("class")
                .setValue("com.google.googlex.cortex.sight.SightTest")
                .build(),
            com.google.protos.sight.x.proto.Sight.Attribute.newBuilder()
                .setKey("function")
                .setValue("logNestedBlockToCapacitorFile")
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
                .setFile("SightTest.java")
                .setLine(baseEventLoc.getLineNumber() + 12)
                .setFunc("logNestedBlockToCapacitorFile")
                .addAllAttribute(attrs)
                .setSubType(SubType.ST_BLOCK_START)
                .setBlockStart(BlockStart.newBuilder().setLabel("outerBlock"))
                .setOrder(Order.newBuilder().setTimestampNs(1000001))
                .build(),
            com.google.protos.sight.x.proto.Sight.Object.newBuilder()
                .setLocation("0000000000:0000000000")
                .setIndex(1)
                .addAncestorStartLocation("0000000000")
                .addAncestorStartLocation("0000000000:0000000000")
                .setFile("SightTest.java")
                .setLine(baseEventLoc.getLineNumber() + 14)
                .setFunc("logNestedBlockToCapacitorFile")
                .addAllAttribute(attrs)
                .setSubType(SubType.ST_TEXT)
                .setText(Text.newBuilder().setText("preText"))
                .setOrder(Order.newBuilder().setTimestampNs(1000002))
                .build(),
            com.google.protos.sight.x.proto.Sight.Object.newBuilder()
                .setLocation("0000000000:0000000001")
                .setIndex(2)
                .addAncestorStartLocation("0000000000")
                .addAncestorStartLocation("0000000000:0000000001")
                .setFile("SightTest.java")
                .setLine(baseEventLoc.getLineNumber() + 15)
                .setFunc("logNestedBlockToCapacitorFile")
                .addAllAttribute(attrs)
                .setSubType(SubType.ST_BLOCK_START)
                .setBlockStart(BlockStart.newBuilder().setLabel("innerBlock"))
                .setOrder(Order.newBuilder().setTimestampNs(1000003))
                .build(),
            com.google.protos.sight.x.proto.Sight.Object.newBuilder()
                .setLocation("0000000000:0000000001:0000000000")
                .setIndex(3)
                .addAncestorStartLocation("0000000000")
                .addAncestorStartLocation("0000000000:0000000001")
                .addAncestorStartLocation("0000000000:0000000001:0000000000")
                .setFile("SightTest.java")
                .setLine(baseEventLoc.getLineNumber() + 16)
                .setFunc("logNestedBlockToCapacitorFile")
                .addAllAttribute(attrs)
                .setSubType(SubType.ST_TEXT)
                .setText(Text.newBuilder().setText("inText"))
                .setOrder(Order.newBuilder().setTimestampNs(1000004))
                .build(),
            com.google.protos.sight.x.proto.Sight.Object.newBuilder()
                .setLocation("0000000000:0000000002")
                .setIndex(4)
                .addAncestorStartLocation("0000000000")
                .addAncestorStartLocation("0000000000:0000000002")
                .setFile("SightTest.java")
                .setLine(baseEventLoc.getLineNumber() + 17)
                .setFunc("logNestedBlockToCapacitorFile")
                .addAllAttribute(attrs)
                .setSubType(SubType.ST_BLOCK_END)
                .setBlockEnd(
                    BlockEnd.newBuilder()
                        .setLabel("innerBlock")
                        .setLocationOfBlockStart("0000000000:0000000001")
                        .setNumDirectContents(1)
                        .setNumTransitiveContents(1)
                        .setMetrics(BlockEnd.Metrics.newBuilder().setElapsedTimeNs(2)))
                .setOrder(Order.newBuilder().setTimestampNs(1000005))
                .build(),
            com.google.protos.sight.x.proto.Sight.Object.newBuilder()
                .setLocation("0000000000:0000000003")
                .setIndex(5)
                .addAncestorStartLocation("0000000000")
                .addAncestorStartLocation("0000000000:0000000003")
                .setFile("SightTest.java")
                .setLine(baseEventLoc.getLineNumber() + 18)
                .setFunc("logNestedBlockToCapacitorFile")
                .addAllAttribute(attrs)
                .setSubType(SubType.ST_TEXT)
                .setText(Text.newBuilder().setText("postText"))
                .setOrder(Order.newBuilder().setTimestampNs(1000006))
                .build(),
            com.google.protos.sight.x.proto.Sight.Object.newBuilder()
                .setLocation("0000000001")
                .setIndex(6)
                .addAncestorStartLocation("0000000001")
                .setFile("SightTest.java")
                .setLine(baseEventLoc.getLineNumber() + 19)
                .setFunc("logNestedBlockToCapacitorFile")
                .addAllAttribute(attrs)
                .setSubType(SubType.ST_BLOCK_END)
                .setBlockEnd(
                    BlockEnd.newBuilder()
                        .setLabel("outerBlock")
                        .setLocationOfBlockStart("0000000000")
                        .setNumDirectContents(3)
                        .setNumTransitiveContents(5)
                        .setMetrics(BlockEnd.Metrics.newBuilder().setElapsedTimeNs(6)))
                .setOrder(Order.newBuilder().setTimestampNs(1000007))
                .build());
  }

  @Test
  public void logAttributesToTextFile() throws CapacitorException, IOException {
    // SET-UP
    Path testDir =
        Files.createDirectories(
            Path.of(tempFolder.newFolder("logAttributesToTextFile").getAbsolutePath()));
    String logFilePath;
    StackTraceElement baseEventLoc = Thread.currentThread().getStackTrace()[1];
    try (SightImpl sight =
            SightImpl.create(
                Params.newBuilder()
                    .setLocal(true)
                    .setTextOutput(true)
                    .setLogDirPath(testDir.toString())
                    .build(),
                /* timeSource= */ FakeTimeSource.create()
                    .setNow(Instant.ofEpochMilli(1))
                    .setAutoAdvance(Duration.ofNanos(1)));
        // ACT
        Attribute a = Attribute.create("key", "val", sight)) {
      logFilePath = sight.getTextLogFilePath().toString();
      sight.textLine("text");
    }
    // ASSERT
    assertThat(FileUtils.readFileToString(new File(logFilePath), UTF_8))
        .isEqualTo(
            String.format(
                "SightTest.java:%s/logAttributesToTextFile text|"
                    + " key=val,runStartTime=1969-12-31T16:00:00.001,taskBNS=\n",
                baseEventLoc.getLineNumber() + 14));
  }

  @Test
  public void logAttributeToCapacitorFile() throws CapacitorException, IOException {
    // SET-UP
    Path testDir =
        Files.createDirectories(
            Path.of(tempFolder.newFolder("logAttributeToCapacitorFile").getAbsolutePath()));
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
                    .setAutoAdvance(Duration.ofNanos(1)));
        // ACT
        Attribute a = Attribute.create("key", "val", sight)) {
      logFilePath = sight.getCapacitorLogFilePath().toString();
      sight.textLine("text");
    }

    ImmutableList<com.google.protos.sight.x.proto.Sight.Attribute> attrs =
        ImmutableList.of(
            com.google.protos.sight.x.proto.Sight.Attribute.newBuilder()
                .setKey("runStartTime")
                .setValue("1969-12-31T16:00:00.001")
                .build(),
            com.google.protos.sight.x.proto.Sight.Attribute.newBuilder()
                .setKey("taskBNS")
                .setValue("")
                .build());

    // ASSERT
    assertThat(loadLog(logFilePath))
        .ignoringFieldDescriptors(IGNORABLE_FIELDS)
        .containsExactly(
            com.google.protos.sight.x.proto.Sight.Object.newBuilder()
                .setLocation("0000000000")
                .addAncestorStartLocation("0000000000")
                .addAttribute(
                    com.google.protos.sight.x.proto.Sight.Attribute.newBuilder()
                        .setKey("key")
                        .setValue("val"))
                .setFile("SightTest.java")
                .setLine(baseEventLoc.getLineNumber() + 14)
                .setFunc("logAttributeToCapacitorFile")
                .addAllAttribute(attrs)
                .setSubType(SubType.ST_TEXT)
                .setText(Text.newBuilder().setText("text\n"))
                .setOrder(Order.newBuilder().setTimestampNs(1000001))
                .build());
  }
}
