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

import static com.google.common.time.ZoneIds.googleZoneId;
import static com.google.thirdparty.grpc.JavaTimeConversions.toGrpcDeadline;
import static java.nio.charset.StandardCharsets.UTF_8;
import static java.util.stream.Collectors.joining;

import com.google.analysis.dremel.core.capacitor.CapacitorException;
import com.google.analysis.dremel.core.capacitor.MessageImporter;
import com.google.borg.util.BorgletInfo;
import com.google.common.flogger.GoogleLogger;
import com.google.common.time.Instants;
import com.google.common.time.TimeSource;
import com.google.common.util.concurrent.Uninterruptibles;
import com.google.devtools.build.runtime.RunfilesPaths;
import com.google.dremel.capacitor.ImporterOptions.MessageImporterOptions;
import com.google.dremel.capacitor.ImporterOptions.MessageImporterOptions.SchemaOptions;
import com.google.errorprone.annotations.ResultIgnorabilityUnspecified;
import com.google.googlex.cortex.sight.sightservice.SightServiceGrpc;
import com.google.googlex.cortex.sight.sightservice.SightServiceGrpc.SightServiceBlockingStub;
import com.google.googlex.cortex.sight.sightservice.SightServiceOuterClass.CreateRequest;
import com.google.googlex.cortex.sight.sightservice.SightServiceOuterClass.CreateResponse;
import com.google.googlex.cortex.sight.sightservice.SightServiceOuterClass.FinalizeRequest;
import com.google.googlex.cortex.sight.sightservice.SightServiceOuterClass.LogFormat;
import com.google.io.file.AccessMode;
import com.google.io.file.GoogleFile;
import com.google.net.grpc.ProdChannelBuilder;
import com.google.protos.sight.x.proto.Sight.Object.SubType;
import com.google.protos.sight.x.proto.Sight.Params;
import com.google.protos.sight.x.proto.Sight.Text;
import com.google.tech.file.TechFiles;
import com.sun.management.OperatingSystemMXBean;
import com.sun.management.ThreadMXBean;
import io.grpc.ManagedChannel;
import java.io.BufferedWriter;
import java.io.IOException;
import java.io.OutputStreamWriter;
import java.lang.management.ManagementFactory;
import java.nio.file.Path;
import java.time.Duration;
import java.time.Instant;
import java.util.ArrayDeque;
import java.util.Map;
import java.util.Optional;
import java.util.TreeMap;
import org.checkerframework.checker.nullness.qual.Nullable;

/** Base implementation of the Sight interface. */
public final class SightImpl implements Sight, AutoCloseable {
  private static final GoogleLogger logger = GoogleLogger.forEnclosingClass();

  /** The object through which we access per-thread CPU & heap utilization. */
  private static final ThreadMXBean threadMXBean;

  static {
    threadMXBean = (ThreadMXBean) ManagementFactory.getThreadMXBean();
    threadMXBean.setThreadAllocatedMemoryEnabled(true);
  }

  /** The object through which we access Operating System metrics. */
  private static final OperatingSystemMXBean osMxBean =
      ManagementFactory.getPlatformMXBean(OperatingSystemMXBean.class);

  /** Creates a Sight object that writes all log data to the standard log. */
  public static SightImpl create() {
    return new SightImpl(/* silentLogger= */ false);
  }

  /** Creates a Sight object that emits no logs. */
  public static SightImpl createSilent() {
    return new SightImpl(/* silentLogger= */ true);
  }

  /** Creates a Sight object configured via {@code params}. */
  public static SightImpl create(Params params) throws CapacitorException, IOException {
    if (params.getLocal()) {
      return create(params, TimeSource.system(), /* channel= */ Optional.empty());
    }
    return create(
        params,
        TimeSource.system(),
        /* channel= */ Optional.of(
            ProdChannelBuilder.forTarget("/bns/wl/borg/wl/bns/bronevet/sight-service.server/0")
                .build()));
  }

  /**
   * Creates a Sight object configured via {@code params} and using {@code timeSource} to measure
   * time.s
   */
  public static SightImpl create(Params params, TimeSource timeSource)
      throws CapacitorException, IOException {
    if (params.getLocal()) {
      return create(params, timeSource, /* channel= */ Optional.empty());
    }
    return create(
        params,
        timeSource,
        /* channel= */ Optional.of(
            ProdChannelBuilder.forTarget("/bns/wl/borg/wl/bns/bronevet/sight-service.server/0")
                .build()));
  }

  /**
   * Creates a Sight object configured via {@code params}, using {@code timeSource} to measure time
   * and using {@code channel} to communicate with the Sight Service.
   */
  public static SightImpl create(
      Params params, TimeSource timeSource, Optional<ManagedChannel> channel)
      throws CapacitorException, IOException {
    SightImpl sight;

    if (params.getContainerLocation().isEmpty()) {
      sight = new SightImpl(/* silentLogger= */ false);
    } else {
      sight = new SightImpl(params.getSilentLogger(), params.getContainerLocation());
    }

    // The path prefix common to all the file(s) that hold the log.
    String logFilePathPrefix;
    if (params.getLocal()) {
      if (!params.getPathPrefix().isEmpty()) {
        logFilePathPrefix = params.getPathPrefix();
      } else {
        String pathLabel = "log";
        if (!params.getLabel().isEmpty()) {
          pathLabel = params.getLabel();
        }
        logFilePathPrefix = String.format("%s/%s", params.getLogDirPath(), pathLabel);
      }
      sight.id = 0;
      sight.sightService = null;
    } else {
      if (channel.isEmpty()) {
        throw new IllegalArgumentException(
            String.format(
                "Cannot create Sight logger in remote mode. Channel (=%s) was not provided",
                channel));
      }
      sight.sightService = SightServiceGrpc.newBlockingStub(channel.get());
      CreateResponse response =
          sight
              .sightService
              .withDeadline(toGrpcDeadline(Duration.ofSeconds(10)))
              .create(
                  CreateRequest.newBuilder()
                      .setLogOwner(params.getLogOwner())
                      .setFormat(LogFormat.LF_CAPACITOR)
                      .build());
      logFilePathPrefix = response.getPathPrefix();
      sight.id = response.getId();
    }

    sight.params = params.toBuilder().setPathPrefix(logFilePathPrefix).build();
    sight.timeSource = timeSource;

    if (!params.getSilentLogger() && params.getTextOutput()) {
      sight.textLogFilePath =
          TechFiles.getPath(
              String.format(
                  "%s.%s-%s.txt",
                  sight.params.getPathPrefix(),
                  sight.params.getContainerLocation().replace(':', '_'),
                  System.identityHashCode(sight)));
      logger.atInfo().log("textLogFilePath=%s", sight.textLogFilePath);
      sight.textLogWriter =
          new BufferedWriter(
              new OutputStreamWriter(
                  GoogleFile.SYSTEM.newOutputStream(
                      sight.textLogFilePath.toString(), AccessMode.WRITE),
                  UTF_8));
    }

    if (!params.getSilentLogger() && params.getCapacitorOutput()) {
      sight.capacitorLogFilePath =
          TechFiles.getPath(
              String.format(
                  "%s.%s-%s.capacitor",
                  sight.params.getPathPrefix(),
                  sight.params.getContainerLocation().replace(':', '_'),
                  System.identityHashCode(sight)));
      sight.capacitorLogWriter =
          new MessageImporter(
              MessageImporterOptions.newBuilder()
                  .setTypeName(
                      com.google.protos.sight.x.proto.Sight.Object.getDescriptor().getFullName())
                  .setSchemaOptions(
                      SchemaOptions.newBuilder()
                          .setEnableGlobalDescriptorDb(false)
                          .setDescriptorDbFilename(
                              RunfilesPaths.getRunfilesDir()
                                  + "/google3/googlex/cortex/sight/proto/sight_proto2db.protodb"))
                  .build());
    }

    sight.pauseLoggingDepth = 0;

    if (BorgletInfo.isRunningUnderBorgletStatic()) {
      sight.setAttribute("taskBNS", BorgletInfo.getInstance().getTaskBns());
    } else {
      sight.setAttribute("taskBNS", "");
    }

    sight.setAttribute("runStartTime", timeSource.now(googleZoneId()).toString());

    return sight;
  }

  private SightImpl(boolean silentLogger) {
    params = Params.newBuilder().setSilentLogger(silentLogger).build();
    location = Location.create();
    index = 0;
    linePrefix = "";
    lineSuffix = "";
    openBlockStartLocations = new ArrayDeque<>();
    numDirectContents = Location.create();
    numTransitiveContents = Location.create();
    activeBlockLabels = new ArrayDeque<>();
    activeBlockStartTime = new ArrayDeque<>();
    attributes = new TreeMap<>();
    pauseLoggingDepth = 0;
  }

  private SightImpl(boolean silentLogger, String serializedLoc) {
    params = Params.newBuilder().setSilentLogger(silentLogger).build();
    location = Location.create(serializedLoc);
    index = 0;
    linePrefix = "";
    lineSuffix = "";
    openBlockStartLocations = new ArrayDeque<>();
    numDirectContents = Location.create();
    numTransitiveContents = Location.create();
    activeBlockLabels = new ArrayDeque<>();
    activeBlockStartTime = new ArrayDeque<>();
    attributes = new TreeMap<>();
    pauseLoggingDepth = 0;
  }

  @Override
  public void close() {
    if (params.getSilentLogger()) {
      return;
    }
    unsetAttribute("runStartTime");
    unsetAttribute("taskBNS");

    if (params.getTextOutput()) {
      try {
        textLogWriter.close();
      } catch (IOException e) {
        logger.atSevere().withCause(e).log("When closing the text file writer.");
      }
    }

    if (params.getCapacitorOutput()) {
      try {
        capacitorLogWriter.finalizeToFile(capacitorLogFilePath.toString());
      } catch (CapacitorException e) {
        logger.atSevere().withCause(e).log("Closing Capacitor log file");
      }
    }

    if (!params.getLocal()) {
      // Wait for all the files to be written out and be visible to Dremel.
      Uninterruptibles.sleepUninterruptibly(Duration.ofSeconds(1));
      sightService
          .withDeadline(toGrpcDeadline(Duration.ofSeconds(10)))
          .finalize(
              FinalizeRequest.newBuilder()
                  .setId(id)
                  .setPathPrefix(params.getPathPrefix())
                  .setLogOwner(params.getLogOwner())
                  .build());
      logger.atInfo().log(
          "Log :"
              + " https://script.google.com/a/google.com/macros/s/AKfycbxf6dqqiVw_ebHK0wNNYSrc0Ga6YtK2zdCSseNN_H4/dev?log_id=%s&log_owner=%s",
          id, params.getLogOwner());
    }
  }

  /** Returns the UUID of this log. */
  public long getId() {
    return id;
  }

  @Override
  public void pauseLogging() {
    ++pauseLoggingDepth;
  }

  @Override
  public void resumeLogging() {
    --pauseLoggingDepth;
  }

  @Override
  public boolean isLoggingEnabled() {
    return pauseLoggingDepth == 0;
  }

  @ResultIgnorabilityUnspecified
  @Override
  public Location text(String text) {
    return text(text, /* locationOfLogEvent= */ Optional.of(Sight.getCallerStackTraceElement()));
  }

  @Override
  public Location text(String text, StackTraceElement locationOfLogEvent) {
    return text(text, Optional.of(locationOfLogEvent));
  }

  private Location text(String text, Optional<StackTraceElement> locationOfLogEvent) {
    if (pauseLoggingDepth > 0) {
      return Location.create();
    }

    return logObject(
        /* advanceLocation= */ true,
        locationOfLogEvent,
        text,
        com.google.protos.sight.x.proto.Sight.Object.newBuilder()
            .setSubType(SubType.ST_TEXT)
            .setText(Text.newBuilder().setText(text.replace("\n", "\\n"))));
  }

  @ResultIgnorabilityUnspecified
  @Override
  public Location textLine(String text) {
    return textLine(
        text, /* locationOfLogEvent= */ Optional.of(Sight.getCallerStackTraceElement()));
  }

  @Override
  public Location textLine(String text, StackTraceElement locationOfLogEvent) {
    return textLine(text, Optional.of(locationOfLogEvent));
  }

  private Location textLine(String text, Optional<StackTraceElement> locationOfLogEvent) {
    if (pauseLoggingDepth > 0) {
      return Location.create();
    }

    String fullTextLine;
    if (locationOfLogEvent.isPresent()) {
      fullTextLine =
          String.format(
              "%s:%d/%s %s%s%s\n",
              locationOfLogEvent.get().getFileName(),
              locationOfLogEvent.get().getLineNumber(),
              locationOfLogEvent.get().getMethodName(),
              linePrefix,
              text,
              lineSuffix);
    } else {
      fullTextLine = String.format("%s%s%s\n", linePrefix, text, lineSuffix);
    }

    return logObject(
        /* advanceLocation= */ true,
        locationOfLogEvent,
        fullTextLine,
        com.google.protos.sight.x.proto.Sight.Object.newBuilder()
            .setSubType(SubType.ST_TEXT)
            .setText(Text.newBuilder().setText((text.replace("\n", "\\n") + "\n"))));
  }

  @Override
  public Location textBlock(String label, String text) {
    Location enterLoc =
        enterBlock(
            label,
            /* locationOfLogEvent= */ Optional.of(Sight.getCallerStackTraceElement()),
            com.google.protos.sight.x.proto.Sight.Object.newBuilder());
    textLine(text);
    exitBlock(label);
    return enterLoc;
  }

  @Override
  public Location enterBlock(String label) {
    return enterBlock(
        label,
        /* locationOfLogEvent= */ Optional.empty(),
        com.google.protos.sight.x.proto.Sight.Object.newBuilder());
  }

  @Override
  public Location enterBlock(
      String label,
      StackTraceElement locationOfLogEvent,
      com.google.protos.sight.x.proto.Sight.Object.Builder object) {
    return enterBlock(label, Optional.of(locationOfLogEvent), object);
  }

  private Location enterBlock(
      String label,
      Optional<StackTraceElement> locationOfLogEvent,
      com.google.protos.sight.x.proto.Sight.Object.Builder object) {
    if (params.getSilentLogger() || pauseLoggingDepth > 0) {
      return location;
    }

    if (locationOfLogEvent.isPresent()) {
      setAttribute("function", locationOfLogEvent.get().getMethodName());
      setAttribute("class", locationOfLogEvent.get().getClassName());
    }

    activeBlockLabels.addLast(label);
    Instant blockStartTime = timeSource.now();
    activeBlockStartTime.addLast(blockStartTime);
    String logText = linePrefix + label + "<<<" + lineSuffix + "\n";
    linePrefix = linePrefix + label + ": ";

    Location objLocation = location;
    object.setSubType(SubType.ST_BLOCK_START);
    object.getBlockStartBuilder().setLabel(label);
    logObject(
        /* advanceLocation= */ false,
        locationOfLogEvent,
        /* currentTime= */ Optional.of(blockStartTime),
        logText,
        object);
    openBlockStartLocations.addLast(object.getLocation());

    numDirectContents.enter(0);
    numTransitiveContents.enter(0);
    location.enter(0);

    return objLocation;
  }

  @Override
  public void exitBlock(String label) {
    exitBlock(
        label,
        /* locationOfLogEvent= */ Optional.empty(),
        com.google.protos.sight.x.proto.Sight.Object.newBuilder());
  }

  @Override
  public void exitBlock(
      String label,
      StackTraceElement locationOfLogEvent,
      com.google.protos.sight.x.proto.Sight.Object.Builder object) {
    exitBlock(label, Optional.of(locationOfLogEvent), object);
  }

  private void exitBlock(
      String label,
      Optional<StackTraceElement> locationOfLogEvent,
      com.google.protos.sight.x.proto.Sight.Object.Builder object) {
    if (params.getSilentLogger() || pauseLoggingDepth > 0) {
      return;
    }

    if (activeBlockLabels.isEmpty() || activeBlockStartTime.isEmpty() || location.size() == 1) {
      logger.atWarning().log("Exiting inactive Sight block \"%s\"", label);
      return;
    }
    Instant blockEndTime = timeSource.now();
    Instant blockStartTime = activeBlockStartTime.removeLast();
    object
        .getBlockEndBuilder()
        .getMetricsBuilder()
        .setElapsedTimeNs(Duration.between(blockStartTime, blockEndTime).toNanos());

    activeBlockLabels.removeLast();
    linePrefix = "";
    for (String blockLabel : activeBlockLabels) {
      linePrefix = linePrefix + blockLabel + ": ";
    }

    location.exit();
    location.next();

      if (openBlockStartLocations.isEmpty()) {
        logger.atWarning().log("Exiting inactive Sight block \"%s\"", label);
        return;
      }

    object.setSubType(SubType.ST_BLOCK_END);
    object
        .getBlockEndBuilder()
        .setLabel(label)
        .setNumDirectContents(numDirectContents.pos())
        .setNumTransitiveContents(numTransitiveContents.pos())
        .setLocationOfBlockStart(openBlockStartLocations.peekLast());
      openBlockStartLocations.removeLast();
    logObject(
        /* advanceLocation= */ true,
        locationOfLogEvent,
        /* currentTime= */ Optional.of(blockEndTime),
        linePrefix + label + ">>>>" + lineSuffix + "\n",
        object);

    numDirectContents.exit();
    numTransitiveContents.exit();

    if (locationOfLogEvent.isPresent()) {
      unsetAttribute("function");
      unsetAttribute("class");
    }
  }

  /** Updates lineSuffix to account for the current state of the log attributes. */
  private void updateLineSuffix() {
    if (attributes.isEmpty()) {
      lineSuffix = "";
    } else {
      lineSuffix =
          "| "
              + attributes.entrySet().stream()
                  .map(attr -> attr.getKey() + "=" + attr.getValue().peekLast())
                  .collect(joining(","));
    }
  }

  @Override
  public void setAttribute(String key, String value) {
    attributes.computeIfAbsent(key, k -> new ArrayDeque<>()).addLast(value);
    updateLineSuffix();
  }

  @Override
  public void unsetAttribute(String key) {
    ArrayDeque<String> values = attributes.get(key);
    if (values == null || values.isEmpty()) {
      logger.atSevere().log("Failed to unset attribute %s, which is not set.", key);
      return;
    }

    values.removeLast();
    if (values.isEmpty()) {
      attributes.remove(key);
    }

    updateLineSuffix();
  }

  @Override
  public String getAttribute(String key) {
    ArrayDeque<String> values = attributes.get(key);
    if (values == null || values.isEmpty()) {
      return "";
    }
    return values.peekLast();
  }

  @Override
  public Location logObject(com.google.protos.sight.x.proto.Sight.Object.Builder object) {
    return logObject(
        /* advanceLocation= */ true,
        /* locationOfLogEvent= */ Optional.empty(),
        /* currentTime= */ Optional.empty(),
        "",
        object);
  }

  @Override
  public Location logObject(
      boolean advanceLocation,
      Optional<StackTraceElement> locationOfLogEvent,
      String text,
      com.google.protos.sight.x.proto.Sight.Object.Builder object) {
    return logObject(
        advanceLocation, locationOfLogEvent, /* currentTime= */ Optional.empty(), text, object);
  }

  @ResultIgnorabilityUnspecified
  @Override
  public Location logObject(
      boolean advanceLocation,
      Optional<StackTraceElement> locationOfLogEvent,
      Optional<Instant> currentTime,
      String text,
      com.google.protos.sight.x.proto.Sight.Object.Builder object) {
    if (params.getSilentLogger()) {
      return Location.create();
    }

    if (!text.isEmpty()) {
      emitTextToFile(text);
    }

    if (!numDirectContents.isEmpty()) {
      numDirectContents.next();
    }
    numTransitiveContents.nextAll();

    Location objLocation = location;
    if (isBinaryLogged()) {
      if (locationOfLogEvent.isPresent()) {
        object
            .setFile(locationOfLogEvent.get().getFileName())
            .setLine(locationOfLogEvent.get().getLineNumber())
            .setFunc(locationOfLogEvent.get().getMethodName());
      }

      object.setLocation(location.toString()).setIndex(index++);
      for (Map.Entry<String, ArrayDeque<String>> attr : attributes.entrySet()) {
        if (attr.getValue().isEmpty()) {
          logger.atSevere().log("No attributes recorded for key %s", attr.getKey());
          continue;
        }
        object.addAttribute(
            com.google.protos.sight.x.proto.Sight.Attribute.newBuilder()
                .setKey(attr.getKey())
                .setValue(attr.getValue().peekLast()));
      }
      object
          .addAllAncestorStartLocation(openBlockStartLocations)
          .addAncestorStartLocation(location.toString());

      object
          .getOrderBuilder()
          .setTimestampNs(
              Instants.toEpochNanosSaturated(currentTime.orElseGet(() -> timeSource.now())));

      object
          .getMetricsBuilder()
          .setProcessFreeSwapSpaceBytes(osMxBean.getFreeSwapSpaceSize())
          .setProcessTotalSwapSpaceBytes(osMxBean.getTotalSwapSpaceSize());

      if (params.getCapacitorOutput()) {
        try {
          capacitorLogWriter.importMessageBytes(object.build().toByteArray());
        } catch (CapacitorException e) {
          logger.atSevere().withCause(e).log("Writing log entry to Capacitor log file");
        }
      }
    }
    if (advanceLocation) {
      location.next();
    }
    return objLocation;
  }

  // Emits text to the output text file, if one is being used.
  private void emitTextToFile(String text) {
    if (params.getSilentLogger()) {
      return;
    }

    if (params.getTextOutput()) {
      try {
        textLogWriter.write(text);
      } catch (IOException e) {
        logger.atSevere().withCause(e).log("Writing text to text log file");
      }
    }
    logger.atInfo().log("%s", text);
  }

  // Returns whether a binary proto representation is being logged.
  private boolean isBinaryLogged() {
    return params.getCapacitorOutput();
  }

  @Override
  public Params getParams() {
    return params;
  }

  public Location getLocation() {
    return location;
  }

  public Path getTextLogFilePath() {
    return textLogFilePath;
  }

  public Path getCapacitorLogFilePath() {
    return capacitorLogFilePath;
  }

  // The configuration parameters of this object.
  private Params params;

  // The TimeSource via which time will be measured.
  private TimeSource timeSource;

  // Connection to the Sight Service
  private @Nullable SightServiceBlockingStub sightService;

  // UUID of this log
  private long id;

  // The file to which the Text representation of the log should be written.
  private Path textLogFilePath;

  // The file to which the Capacitor representation of the log should be
  // written.
  private Path capacitorLogFilePath;

  // If an output file path is specified, this writer is used to emit
  // the log in proto format.
  private BufferedWriter textLogWriter;

  // If an output file path is specified, this writer is used to emit
  // the log in Capacitor format.
  private MessageImporter capacitorLogWriter;

  // The current location in the nesting hierarchy of the log.
  private final Location location;

  // The index of the next log object.
  private long index;

  // The string to be prepended to every log line.
  private String linePrefix;

  // The string to be appended to every log line.
  private String lineSuffix;

  // The stack of locations of all the blocks that have been entered but not
  // yet exited.
  private final ArrayDeque<String> openBlockStartLocations;

  // The number of objects directly contained by currently active block
  // (organized as a stack with one level per block).
  private final Location numDirectContents;

  // The number of objects transitively contained by currently active block
  // (organized as a stack with one level per block).
  private final Location numTransitiveContents;

  // The stack of the labels of blocks that are currently active (begun but not
  // yet ended).
  private final ArrayDeque<String> activeBlockLabels;

  // The stack of starting timestamps of all blocks that are currently active (begun but not yet
  // ended).
  private final ArrayDeque<Instant> activeBlockStartTime;

  // The key-value attribute mappings that currently in-force. Maps each key to
  // the stack of key-value mappings in the order they were set.
  private final TreeMap<String, ArrayDeque<String>> attributes;

  // Counts the numer of PauseLogging() calls with no matching calls to
  // ResumeLogging();
  private int pauseLoggingDepth;
}
