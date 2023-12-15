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

"""Tests for google3.googlex.cortex.sight.py.location."""

from sight import Location
from absl.testing import absltest


class LocationTest(absltest.TestCase):

  def testDefaultLocation(self):
    # SETUP
    loc = Location()

    # ASSERT
    self.assertEqual(loc.size(), 1)
    self.assertEqual(loc.pos(), 0)
    self.assertEqual(loc.is_empty(), False)
    self.assertEqual(str(loc), '0000000000')

  def testEnterExitEmptyLocation(self):
    # SETUP
    loc = Location()

    # ACT
    loc.enter(1)
    loc.exit()
    loc.exit()

    # ASSERT
    self.assertEqual(loc.is_empty(), True)
    self.assertEqual(str(loc), '')

  def testEnterExitLocation(self):
    # SETUP
    loc = Location()

    for i in range(1, 5):
      # ACT
      loc.enter(i)

      # ASSERT
      self.assertEqual(loc.size(), i + 1)
      self.assertEqual(loc.pos(), i)

    # ASSERT
    self.assertEqual(
        str(loc), '0000000000:0000000001:0000000002:0000000003:0000000004'
    )

    for i in range(4, 0, -1):
      # ACT
      loc.exit()

      # ASSERT
      self.assertEqual(loc.size(), i)
      self.assertEqual(loc.pos(), i - 1)

    # ASSERT
    self.assertEqual(str(loc), '0000000000')

  def testNextLocation(self):
    # SETUP
    loc = Location()

    for i in range(1, 5):
      # ACT
      loc.enter(i)
      for j in range(1, 5):
        loc.next()

        # ASSERT
        self.assertEqual(loc.pos(), i + j)

    # ASSERT
    self.assertEqual(
        str(loc), '0000000000:0000000005:0000000006:0000000007:0000000008'
    )

  def testNextAllLocation(self):
    # SETUP
    loc = Location()

    # ACT
    for i in range(1, 5):
      loc.enter(i)

    loc.next_all()

    # ASSERT
    self.assertEqual(
        str(loc), '0000000001:0000000002:0000000003:0000000004:0000000005'
    )

    for i in range(4, 0, -1):
      # ACT
      loc.exit()

      # ASSERT
      self.assertEqual(loc.pos(), i)

    # ASSERT
    self.assertEqual(str(loc), '0000000001')


if __name__ == '__main__':
  absltest.main()
