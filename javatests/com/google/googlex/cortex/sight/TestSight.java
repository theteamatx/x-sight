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

import com.google.analysis.dremel.core.capacitor.CapacitorException;
import com.google.net.loas.Loas;
import com.google.protos.sight.x.proto.Sight.Params;
import com.google.testing.junit.runner.util.CurrentRunningTest;
import java.io.IOException;
import java.util.Optional;
import org.junit.rules.ExternalResource;

/**
 * Java Test rule that makes Sight available to JUnit tests and organizes the resulting
 * logs based on the test classes and methods that emitted them.
 */
public final class TestSight extends ExternalResource {
  Optional<SightImpl> fullTestSight = Optional.empty();
  Optional<SightImpl> sight;

  @Override
  protected void before() {
    // If running before all tests.
    if (CurrentRunningTest.get() == null) {
      try {
        fullTestSight =
            Optional.of(
                SightImpl.create(
                    Params.newBuilder()
                        .setLogOwner(Loas.getDefaultUserName())
                        .setCapacitorOutput(true)
                        .build()));
      } catch (CapacitorException | IOException e) {
        sight = Optional.empty();
      }
      // If running before a specific test.
    } else {
      if (fullTestSight.isPresent()) {
        try {
          sight =
              Optional.of(
                  SightImpl.create(
                      fullTestSight.get().getParams().toBuilder()
                          .setLocal(true)
                          .setContainerLocation(
                              fullTestSight
                                  .get()
                                  .enterBlock(CurrentRunningTest.get().getDisplayName())
                                  .toString())
                          .build()));
          sight.get().getLocation().enter(sight.hashCode());
          sight.get().setAttribute("Test Class", CurrentRunningTest.get().getClassName());
          sight.get().setAttribute("Test Method", CurrentRunningTest.get().getMethodName());
          fullTestSight.get().exitBlock(CurrentRunningTest.get().getDisplayName());
        } catch (CapacitorException | IOException e) {
          sight = Optional.empty();
        }
      }
    }
  }

  public Sight get() {
    return sight.get();
  }

  @Override
  protected void after() {
    // If running at the end of the entire test suite.
    if (CurrentRunningTest.get() == null) {
      if (fullTestSight.isPresent()) {
        fullTestSight.get().close();
      }
      // If running after a single test.
    } else {
      if (sight.isPresent()) {
        sight.get().unsetAttribute("Test Method");
        sight.get().unsetAttribute("Test Class");
        sight.get().getLocation().exit();
        sight.get().close();
      }
    }
  }
}
