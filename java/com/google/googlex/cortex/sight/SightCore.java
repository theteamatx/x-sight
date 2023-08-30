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

import com.google.errorprone.annotations.ResultIgnorabilityUnspecified;
import com.google.protos.sight.x.proto.Sight.Params;
import java.time.Instant;
import java.util.Optional;

/**
 * Specifies a simplified version of the generic interface to structured logging that provides basic
 * logging functionality without access to any of the widgets. It is this directly usable by the
 * widgets themselves.
 */
public interface SightCore {
  /** Returns the configuration parameters of this Sight instance. */
  Params getParams();

  /**
   * Specifies that logging should not be performed until the next call to ResumeLogging(). If
   * PauseLogging() is called multiple times, logging is not resumed until ResumeLogging() is called
   * once for each call to PauseLogging().
   */
  public void pauseLogging();

  /** Resumes logging after a call to PauseLogging(). */
  public void resumeLogging();

  /** Returns whether logging is currently enabled. */
  public boolean isLoggingEnabled();

  /** Emits to the log a line containing {@code text}. */
  public Location text(String text);

  /**
   * Emits to the log a line containing {@code text}.
   *
   * @param locationOfLogEvent indicates the code location where the logging event occurred.
   */
  public Location text(String text, StackTraceElement locationOfLogEvent);

  /** Emits to the log a line containing {@code text}, followed by a linebreak. */
  @ResultIgnorabilityUnspecified
  public Location textLine(String text);

  /**
   * Emits to the log a line containing {@code text}, followed by a linebreak.
   *
   * @param locationOfLogEvent indicates the code location where the logging event occurred.
   */
  public Location textLine(String text, StackTraceElement locationOfLogEvent);

  /**
   * Enters a new block.
   *
   * @param label the name of the block
   * @return the location of the emitted log object.
   */
  public Location enterBlock(String label);

  /**
   * Enters a new block.
   *
   * @param label the name of the block
   * @param locationOfLogEvent indicates the code location where the logging event occurred.
   * @param object describes an object that inherits from StartBlock, with all of its specialized
   *     fields set.
   * @return the location of the emitted log object.
   */
  @ResultIgnorabilityUnspecified
  public Location enterBlock(
      String label,
      StackTraceElement locationOfLogEvent,
      com.google.protos.sight.x.proto.Sight.Object.Builder object);

  /** Exits a previously-entered block named label. */
  public void exitBlock(String label);

  /**
   * Exits a previously-entered block.
   *
   * @param label the name of the entered block
   * @param locationOfLogEvent indicates the code location where the logging event occurred.
   * @param object describes an object that inherits from EndBlock, with all of its specialized
   *     fields set.
   */
  public void exitBlock(
      String label,
      StackTraceElement locationOfLogEvent,
      com.google.protos.sight.x.proto.Sight.Object.Builder object);

  /**
   * Marks the beginning of a log region where key is mapped to value val. If key is currently set
   * to some value, the new mapping supersedes the prior one until the next call to UnsetAttribute()
   * with the same key.
   */
  public void setAttribute(String key, String value);

  /**
   * Marks the end of a log region where key is mapped to value val. If at the time of the last call
   * to SetAttribute() with key, key was already mapped to a value, the Sight reverts to using that
   * mapping.
   */
  public void unsetAttribute(String key);

  /** Returns the value currently mapped to key or an empty string if key is currently unmapped. */
  public String getAttribute(String key);

  /** Emits object to the log. Returns the location of the emitted log object. */
  public Location logObject(com.google.protos.sight.x.proto.Sight.Object.Builder object);

  /**
   * Emits to the text log {@code objText} (if text logging is enabled) to the binary log {@code
   * object} (if binary logging is enabled).
   *
   * @return the location of the emitted log object.
   * @param advanceLocation Indicates whether this method call should advance the current log
   *     location.
   * @param locationOfLogEvent indicates the code location where the logging event occurred.
   * @param object describes an object that inherits from StartBlock, with all of its specialized
   *     fields set.
   */
  public Location logObject(
      boolean advanceLocation,
      Optional<StackTraceElement> locationOfLogEvent,
      String text,
      com.google.protos.sight.x.proto.Sight.Object.Builder object);

  /**
   * Emits to the text log {@code objText} (if text logging is enabled) to the binary log {@code
   * object} (if binary logging is enabled).
   *
   * @return the location of the emitted log object.
   * @param advanceLocation Indicates whether this method call should advance the current log
   *     location.
   * @param locationOfLogEvent indicates the code location where the logging event occurred.
   * @param currentTime the time when the event was logged.
   * @param object describes an object that inherits from StartBlock, with all of its specialized
   *     fields set.
   */
  public Location logObject(
      boolean advanceLocation,
      Optional<StackTraceElement> locationOfLogEvent,
      Optional<Instant> currentTime,
      String text,
      com.google.protos.sight.x.proto.Sight.Object.Builder object);
}
