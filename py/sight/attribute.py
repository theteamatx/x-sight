# Copyright 2023 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Hierarchical attribute annotations in the sight log."""

from sight.sight import Sight


class Attribute(object):
  """Encapsulates start and stop points where an attribute is set to a value."""

  def __init__(self, key: str, val: str, sight: Sight):
    self.key = key
    self.sight = sight
    self.sight.set_attribute(self.key, val)

  def __enter__(self):
    return self

  def __exit__(self, exc_type, value, traceback):
    self.sight.unset_attribute(self.key)
