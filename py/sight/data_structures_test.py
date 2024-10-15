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
"""Tests for py.sight."""

import inspect

from absl.testing import absltest
from google3.analysis.dremel.core.capacitor.public.python import (
    pywrap_record_reader
)
import numpy as np
from sight.proto import sight_pb2
from sight.py import data_structures
from sight.sight import Sight


class DataStructuresTest(absltest.TestCase):

  @staticmethod
  def _read_capacitor_file(file_path: str):
    protos = []
    record_reader = pywrap_record_reader.RecordReader.CreateFromPath(
        file_path, ['*'], 60.0)
    protos.extend(record_reader.IterRecords())
    return sorted(protos, key=lambda x: x.index)

  def testBaseTypeLogging(self):
    # SETUP
    params = sight_pb2.Params()
    params.label = 'testBaseTypeLogging'
    params.log_owner = 'mdb/sight-dev'
    params.local = True
    params.capacitor_output = True
    params.log_dir_path = self.create_tempdir().full_path

    # ACT
    frame = inspect.currentframe()
    with Sight(params) as sight:
      data_structures.log('abc', sight, frame)
      data_structures.log(b'xyz', sight, frame)
      data_structures.log(5, sight, frame)
      data_structures.log(2.5, sight, frame)
      data_structures.log(True, sight, frame)
      data_structures.log(None, sight, frame)

    # ASSERT
    actual_log = self._read_capacitor_file(params.log_dir_path +
                                           '/testBaseTypeLogging.capacitor')

    self.assertEqual('abc', data_structures.from_log([actual_log[0]]))
    self.assertEqual(b'xyz', data_structures.from_log([actual_log[1]]))
    self.assertEqual(5, data_structures.from_log([actual_log[2]]))
    self.assertEqual(2.5, data_structures.from_log([actual_log[3]]))
    self.assertEqual(True, data_structures.from_log([actual_log[4]]))
    self.assertIsNone(data_structures.from_log([actual_log[5]]))

  def testSequenceLogging(self):
    # SETUP
    params = sight_pb2.Params()
    params.label = 'testSequenceLogging'
    params.log_owner = 'mdb/sight-dev'
    params.local = True
    params.capacitor_output = True
    params.log_dir_path = self.create_tempdir().full_path

    # ACT
    frame = inspect.currentframe()
    with Sight(params) as sight:
      data_structures.log(['abc', b'xyz', 5, 2.5, True, None], sight, frame)
      data_structures.log(('abc', b'xyz', 5, 2.5, True, None), sight, frame)
      data_structures.log({'abc', b'xyz', 5, 2.5, True, None}, sight, frame)

    # ASSERT
    actual_log = self._read_capacitor_file(params.log_dir_path +
                                           '/testSequenceLogging.capacitor')

    self.assertEqual(
        ['abc', b'xyz', 5, 2.5, True, None],
        data_structures.from_log(list(actual_log[0:8])),
    )
    self.assertEqual(
        ('abc', b'xyz', 5, 2.5, True, None),
        data_structures.from_log(list(actual_log[8:16])),
    )
    self.assertEqual(
        {'abc', b'xyz', 5, 2.5, True, None},
        data_structures.from_log(list(actual_log[16:24])),
    )

  def testDictLogging(self):
    # SETUP
    params = sight_pb2.Params()
    params.label = 'testDictLogging'
    params.log_owner = 'mdb/sight-dev'
    params.local = True
    params.capacitor_output = True
    params.log_dir_path = self.create_tempdir().full_path

    # ACT
    frame = inspect.currentframe()
    with Sight(params) as sight:
      data_structures.log({'abc': b'xyz', 5: 2.5, True: None}, sight, frame)

    # ASSERT
    actual_log = self._read_capacitor_file(params.log_dir_path +
                                           '/testDictLogging.capacitor')

    self.assertEqual(
        {
            'abc': b'xyz',
            5: 2.5,
            True: None
        },
        data_structures.from_log(actual_log),
    )

  def testNamedValueDictLogging(self):
    # SETUP
    params = sight_pb2.Params()
    params.label = 'testNamedValueDictLogging'
    params.log_owner = 'mdb/sight-dev'
    params.local = True
    params.capacitor_output = True
    params.log_dir_path = self.create_tempdir().full_path

    # ACT
    frame = inspect.currentframe()
    with Sight(params) as sight:
      data_structures.log_var('map', {
          'abc': b'xyz',
          5: 2.5,
          True: None
      }, sight, frame)

    # ASSERT
    actual_log = self._read_capacitor_file(
        params.log_dir_path + '/testNamedValueDictLogging.capacitor')

    self.assertEqual(
        ('map', {
            'abc': b'xyz',
            5: 2.5,
            True: None
        }),
        data_structures.from_log(actual_log),
    )

  def testTensorInt64Logging(self):
    # SETUP
    params = sight_pb2.Params()
    params.label = 'testTensorInt64Logging'
    params.log_owner = 'mdb/sight-dev'
    params.local = True
    params.capacitor_output = True
    params.log_dir_path = self.create_tempdir().full_path

    # ACT
    frame = inspect.currentframe()
    with Sight(params) as sight:
      data_structures.log(np.array([[1, 2], [3, 4], [5, 6]]), sight, frame)

    # ASSERT
    actual_log = self._read_capacitor_file(params.log_dir_path +
                                           '/testTensorInt64Logging.capacitor')

    np.testing.assert_array_almost_equal(np.array([[1, 2], [3, 4], [5, 6]]),
                                         data_structures.from_log(actual_log))

  def testTensorDoubleLogging(self):
    # SETUP
    params = sight_pb2.Params()
    params.label = 'testTensorDoubleLogging'
    params.log_owner = 'mdb/sight-dev'
    params.local = True
    params.capacitor_output = True
    params.log_dir_path = self.create_tempdir().full_path

    # ACT
    frame = inspect.currentframe()
    with Sight(params) as sight:
      data_structures.log(np.array([[1.1, 2.2], [3.3, 4.4], [5.5, 6.6]]), sight,
                          frame)

    # ASSERT
    actual_log = self._read_capacitor_file(params.log_dir_path +
                                           '/testTensorDoubleLogging.capacitor')

    np.testing.assert_array_almost_equal(
        np.array([[1.1, 2.2], [3.3, 4.4], [5.5, 6.6]]),
        data_structures.from_log(actual_log),
    )


if __name__ == '__main__':
  absltest.main()
