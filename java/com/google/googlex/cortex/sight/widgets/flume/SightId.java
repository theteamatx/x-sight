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

package com.google.googlex.cortex.sight.widgets.flume;

import com.google.auto.value.AutoValue;
import com.google.common.reflect.TypeToken;
import com.google.pipeline.flume.fj.FlumeAutoValue;
import com.google.pipeline.flume.fj.PDataType;
import com.google.pipeline.flume.fj.TypeInference;

/** Unique id that identifies a single execution of a process method of a Flume stage. */
@AutoValue
@FlumeAutoValue
public abstract class SightId {
  // Unique id of the Flume stage.
  public abstract long inputStageId();

  // Unique id of the item within the stage.
  public abstract long inputItemId();

  public static SightId zero() {
    return new AutoValue_SightId(/* inputStageId= */ 0L, /* inputItemId= */ 0L);
  }

  public static SightId create(long inputStageId, long inputItemId) {
    return new AutoValue_SightId(inputStageId, inputItemId);
  }

  public static PDataType<SightId> getPDataType() {
    PDataType<SightId> dataType = new TypeInference().inferPDataType(new TypeToken<SightId>() {});
    return dataType;
  }
}
