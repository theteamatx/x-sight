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

/** Defines the functionality required to make an object loggable by Sight. */
public interface SightLoggable {
  /**
   * Logs this object to {@code sight}, setting the calling location of the logging event as
   * appropriate.
   *
   * @return The log location of the logged object.
   */
  Location log(Sight sight);

  /**
   * Logs this object to {@code sight} at event calling location {@code locationOfLogEvent}.
   *
   * @return The log location of the logged object.
   */
  Location log(StackTraceElement locationOfLogEvent, Sight sight);
}
