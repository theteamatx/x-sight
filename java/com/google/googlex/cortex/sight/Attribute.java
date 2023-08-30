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

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import java.util.Map;
import org.checkerframework.checker.nullness.qual.Nullable;

/**
 * Class that makes it easy to match the lifetimes of attributes in the log to lexical scopes in the
 * source code since they will match the lifetime of Attribute-type variable objects.
 */
public class Attribute implements AutoCloseable {
  private ImmutableList<String> attributeKeys;

  private Sight sight;

  public static @Nullable Attribute create(String key, String val, @Nullable Sight sight) {
    return create(ImmutableMap.of(key, val), sight);
  }

  public static @Nullable Attribute create(Map<String, String> attributes, @Nullable Sight sight) {
    if (sight == null) {
      return null;
    }
    Attribute a = new Attribute();
    a.attributeKeys = ImmutableList.copyOf(attributes.keySet());
    a.sight = sight;
    for (String key : a.attributeKeys) {
      sight.setAttribute(key, attributes.get(key));
    }
    return a;
  }

  @Override
  public void close() {
    for (String key : attributeKeys.reverse()) {
      sight.unsetAttribute(key);
    }
  }
}
