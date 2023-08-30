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

"""Unique IDs for locations within the Sight log."""


class Location(object):
  """Tracks unique IDs of hierarchical blocks and their contents."""

  def __init__(self):
    self.id = [0]

  def enter(self, deeper_id: int) -> None:
    """Enters a next deeper level of the hierarchy.

    Args:
      deeper_id: The starting counter value of the deeper level.
    """
    self.id.append(deeper_id)

  def exit(self) -> None:
    """Exits the current level of the hierarchy."""
    self.id.pop()

  def next(self) -> None:
    """Advances to the next position at the current level of the hierarchy."""
    self.id[-1] += 1

  def next_all(self) -> None:
    """Advances all levels of the hierarchy forward 1 step."""
    for i in range(len(self.id)):
      self.id[i] += 1

  def pos(self) -> int:
    """Returns the current position within the deepest hierarchy level."""
    return self.id[-1]

  def is_empty(self) -> bool:
    """Returns whether the currently location has entered the hierarchy."""
    return self.size() == 0

  def size(self) -> int:
    """Returns the number of hierarchy levels that are currently entered."""
    return len(self.id)

  def __str__(self):
    return ':'.join([str(c).zfill(10) for c in self.id])
