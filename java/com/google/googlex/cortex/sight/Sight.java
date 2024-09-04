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

import com.google.errorprone.annotations.ResultIgnorabilityUnspecified;

/**
 * Specifies a complete version of the generic interface to structured logging that basic logging
 * functionality as well as access to any of the widgets.
 */
public interface Sight extends SightCore {
  /**
   * Emits a block with {@code label} that contains {@code text}.
   *
   * @return the location of the emitted block entry log object.
   */
  public Location textBlock(String label, String text);

  /**
   * Specifies that logging should not be performed until the next call to ResumeLogging(). If
   * PauseLogging() is called multiple times, logging is not resumed until ResumeLogging() is called
   * once for each call to PauseLogging().
   */
  public static void pauseLogging(Sight sight) {
    if (sight != null) {
      sight.pauseLogging();
    }
  }

  /** Resumes logging after a call to PauseLogging(). */
  public static void resumeLogging(Sight sight) {
    if (sight != null) {
      sight.resumeLogging();
    }
  }

  /** Returns whether logging is currently enabled. */
  public static boolean isLoggingEnabled(Sight sight) {
    if (sight != null) {
      return sight.isLoggingEnabled();
    }
    return false;
  }

  /** Emits to the log a line containing text. */
  @ResultIgnorabilityUnspecified
  public static Location text(String text, Sight sight) {
    if (sight != null) {
      return sight.text(text, getCallerStackTraceElement());
    }
    return Location.create();
  }

  /** Emits to the log a line containing text, followed by a linebreak. */
  @ResultIgnorabilityUnspecified
  public static Location textLine(String text, Sight sight) {
    if (sight != null) {
      return sight.textLine(text, getCallerStackTraceElement());
    }
    return Location.create();
  }

  /** Finishes writing this log and ensures that it is available to be read. */
  public void close();

  /**
   * Returns the stack frame with {@code callStack} that corresponds to the function that called the
   * immediate caller of this method.
   */
  public static StackTraceElement getCallerStackTraceElement() {
    // The [0] stack trace element is inside getStackTrace, [1] is this call to
    // getCallerStackTraceElement(), [2] is the calling function. Return [3] element to capture
    // the immediate caller of the function calling getCallerStackTraceElement().
    return Thread.currentThread().getStackTrace()[3];
  }
}
