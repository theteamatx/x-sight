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
from typing import Sequence
# from google3.analysis.dremel.core.capacitor.public.python import pywrap_record_reader
from sight.proto import sight_pb2
from sight.attribute import Attribute
from sight.block import Block
from sight.sight import Sight
from sight.sight import text
from tensorflow.python.util.protobuf import compare
from absl.testing import absltest


class SightTest(absltest.TestCase):

  @classmethod
  def setUpClass(cls):
    super().setUpClass()
    cls.test_path = 'googlex/cortex/sight/py/sight_test.py'

  @staticmethod
  def _read_text_file(file_path: str):
    read_text = ''
    with open(file_path, encoding='UTF-8') as fp:
      for line in fp:
        read_text += line
    return read_text

  # @staticmethod
  # def _read_capacitor_file(file_path: str):
  #   protos = []
  #   record_reader = pywrap_record_reader.RecordReader.CreateFromPath(
  #       file_path, ['*'], 60.0)
  #   protos.extend(record_reader.IterRecords())
  #   return sorted(protos, key=lambda x: x.index)
  @staticmethod
  def _create_attributes(
      base_attributes: Sequence[sight_pb2.Attribute], sight: Sight
  ) -> Sequence[sight_pb2.Attribute]:
    attribute = []
    if hasattr(sight, 'change_list_number'):
      attribute.append(
          sight_pb2.Attribute(
              key='change_list_number', value=str(sight.change_list_number)
          )
      )
    if hasattr(sight, 'citc_snapshot'):
      attribute.append(
          sight_pb2.Attribute(
              key='citc_snapshot', value=str(sight.citc_snapshot)
          )
      )
    attribute.extend(base_attributes)
    return attribute

  @staticmethod
  def _create_attributes_text(
      base_attributes: Sequence[sight_pb2.Attribute], sight: Sight
  ) -> str:
    attribute = []
    if hasattr(sight, 'change_list_number'):
      attribute.append(f'change_list_number={sight.change_list_number}')
    if hasattr(sight, 'citc_snapshot'):
      attribute.append(f'citc_snapshot={sight.citc_snapshot}')
    attribute.extend([f'{a.key}={a.value}' for a in base_attributes])
    return ','.join(attribute)

  def testLogTextToTextFile(self):
    # SETUP
    params = sight_pb2.Params()
    params.label = 'testLogTextToTextFile'
    params.log_owner = 'mdb/x-pitchfork'
    params.local = True
    params.text_output = True
    params.log_dir_path = self.create_tempdir().full_path
    # ACT
    frameinfo = inspect.getframeinfo(inspect.currentframe())
    with Sight(params) as sight:
      sight.text('textA', end='')
      text('textB', sight, end='')
      sight.text('textLine1')
      text('textLine2', sight)
      # ASSERT
      block_attrs = '| ' + self._create_attributes_text([], sight)
    expected_log = """textAtextB%s:%d/%s textLine1%s
%s:%d/%s textLine2%s\n""" % (
        self.test_path,
        frameinfo.lineno + 4,
        frameinfo.function,
        block_attrs,
        self.test_path,
        frameinfo.lineno + 5,
        frameinfo.function,
        block_attrs,
    )
    actual_log = self._read_text_file(
        params.log_dir_path + '/testLogTextToTextFile.txt'
    )
    self.assertEqual(
        expected_log,
        actual_log,
        'Target code and generated logs are different. Expected log:\n%s\n'
        'Actual log:\n%s\n' % (expected_log, actual_log),
    )

  def testLogTextToCapacitorFile(self):
    # SETUP
    params = sight_pb2.Params()
    params.label = 'testLogTextToCapacitorFile'
    params.log_owner = 'mdb/x-pitchfork'
    params.local = True
    params.capacitor_output = True
    params.log_dir_path = self.create_tempdir().full_path
    # ACT
    frameinfo = inspect.getframeinfo(inspect.currentframe())
    with Sight(params) as sight:
      sight.text('textA', end='')
      text('textB', sight, end='')
      sight.text('textLine1')
      text('textLine2', sight)
    # ASSERT
    expected_log = [
        sight_pb2.Object(
            location='0000000000',
            index=0,
            attribute=self._create_attributes([], sight),
            file=self.test_path,
            line=frameinfo.lineno + 2,
            func=frameinfo.function,
            ancestor_start_location=['0000000000'],
            sub_type=sight_pb2.Object.ST_TEXT,
            text=sight_pb2.Text(sub_type=sight_pb2.Text.ST_TEXT, text='textA'),
        ),
        sight_pb2.Object(
            location='0000000001',
            index=1,
            attribute=self._create_attributes([], sight),
            file=self.test_path,
            line=frameinfo.lineno + 3,
            func=frameinfo.function,
            ancestor_start_location=['0000000001'],
            sub_type=sight_pb2.Object.ST_TEXT,
            text=sight_pb2.Text(sub_type=sight_pb2.Text.ST_TEXT, text='textB'),
        ),
        sight_pb2.Object(
            location='0000000002',
            index=2,
            attribute=self._create_attributes([], sight),
            file=self.test_path,
            line=frameinfo.lineno + 4,
            func=frameinfo.function,
            ancestor_start_location=['0000000002'],
            sub_type=sight_pb2.Object.ST_TEXT,
            text=sight_pb2.Text(
                sub_type=sight_pb2.Text.ST_TEXT, text='textLine1\n'
            ),
        ),
        sight_pb2.Object(
            location='0000000003',
            index=3,
            attribute=self._create_attributes([], sight),
            file=self.test_path,
            line=frameinfo.lineno + 5,
            func=frameinfo.function,
            ancestor_start_location=['0000000003'],
            sub_type=sight_pb2.Object.ST_TEXT,
            text=sight_pb2.Text(
                sub_type=sight_pb2.Text.ST_TEXT, text='textLine2\n'
            ),
        ),
    ]
    actual_log = self._read_capacitor_file(
        params.log_dir_path + '/testLogTextToCapacitorFile.capacitor'
    )
    self.assertEqual(len(expected_log), len(actual_log))
    for i in range(0, len(expected_log)):
      compare.assertProtoEqual(
          self,
          expected_log[i],
          actual_log[i],
          'Target code and generated logs are different. Expected'
          ' log[%d]:\n%s\nActual log[%d]:\n%s\n'
          % (i, expected_log[i], i, actual_log[i]),
      )

  def testLogBlockTextToTextFile(self):
    # SETUP
    params = sight_pb2.Params()
    params.label = 'testLogBlockTextToTextFile'
    params.log_owner = 'mdb/x-pitchfork'
    params.local = True
    params.text_output = True
    params.log_dir_path = self.create_tempdir().full_path
    # ACT
    frameinfo = inspect.getframeinfo(inspect.currentframe())
    with Sight(params) as sight:
      with Block('block', sight):
        text('text', sight)
    # ASSERT
    block_attrs = '| ' + self._create_attributes_text([], sight)
    expected_log = """block<<<%s
%s:%d/%s block: text%s
block>>>%s
""" % (
        block_attrs,
        self.test_path,
        frameinfo.lineno + 3,
        frameinfo.function,
        block_attrs,
        block_attrs,
    )
    actual_log = self._read_text_file(
        params.log_dir_path + '/testLogBlockTextToTextFile.txt'
    )
    self.assertEqual(
        expected_log,
        actual_log,
        'Target code and generated logs are different. Expected log:\n%s\n'
        'Actual log:\n%s\n' % (expected_log, actual_log),
    )

  def testLogBlockTextToCapacitorFile(self):
    # SETUP
    params = sight_pb2.Params()
    params.label = 'testLogBlockTextToCapacitorFile'
    params.log_owner = 'mdb/x-pitchfork'
    params.local = True
    params.capacitor_output = True
    params.log_dir_path = self.create_tempdir().full_path
    # ACT
    frameinfo = inspect.getframeinfo(inspect.currentframe())
    with Sight(params) as sight:
      with Block('block', sight):
        text('text', sight)
    # ASSERT
    expected_log = [
        sight_pb2.Object(
            location='0000000000',
            index=0,
            attribute=self._create_attributes([], sight),
            file=self.test_path,
            line=frameinfo.lineno + 2,
            func=frameinfo.function,
            ancestor_start_location=['0000000000'],
            sub_type=sight_pb2.Object.ST_BLOCK_START,
            block_start=sight_pb2.BlockStart(label='block'),
        ),
        sight_pb2.Object(
            location='0000000000:0000000000',
            index=1,
            attribute=self._create_attributes([], sight),
            file=self.test_path,
            line=frameinfo.lineno + 3,
            func=frameinfo.function,
            ancestor_start_location=['0000000000', '0000000000:0000000000'],
            sub_type=sight_pb2.Object.ST_TEXT,
            text=sight_pb2.Text(sub_type=sight_pb2.Text.ST_TEXT, text='text\n'),
        ),
        sight_pb2.Object(
            location='0000000001',
            index=2,
            attribute=self._create_attributes([], sight),
            file=self.test_path,
            line=frameinfo.lineno + 3,
            func=frameinfo.function,
            ancestor_start_location=['0000000001'],
            sub_type=sight_pb2.Object.ST_BLOCK_END,
            block_end=sight_pb2.BlockEnd(
                label='block',
                location_of_block_start='0000000000',
                num_direct_contents=1,
                num_transitive_contents=1,
            ),
        ),
    ]
    actual_log = self._read_capacitor_file(
        params.log_dir_path + '/testLogBlockTextToCapacitorFile.capacitor'
    )
    self.assertEqual(len(expected_log), len(actual_log))
    for i in range(0, len(expected_log)):
      compare.assertProtoEqual(
          self,
          expected_log[i],
          actual_log[i],
          'Target code and generated logs are different. Expected'
          ' log[%d]:\n%s\nActual log[%d]:\n%s\n'
          % (i, expected_log[i], i, actual_log[i]),
      )

  def testLogNestedBlockTextToTextFile(self):
    # SETUP
    params = sight_pb2.Params()
    params.label = 'testLogNestedBlockTextToTextFile'
    params.log_owner = 'mdb/x-pitchfork'
    params.local = True
    params.text_output = True
    params.log_dir_path = self.create_tempdir().full_path
    # ACT
    frameinfo = inspect.getframeinfo(inspect.currentframe())
    with Sight(params) as sight:
      with Block('outerBlock', sight):
        text('preText', sight)
        with Block('innerBlock', sight):
          text('inText', sight)
        text('postText', sight)
    # ASSERT
    block_attrs = '| ' + self._create_attributes_text([], sight)
    expected_log = """outerBlock<<<%s
%s:%d/%s outerBlock: preText%s
outerBlock: innerBlock<<<%s
%s:%d/%s outerBlock: innerBlock: inText%s
outerBlock: innerBlock>>>%s
%s:%d/%s outerBlock: postText%s
outerBlock>>>%s
""" % (
        block_attrs,
        self.test_path,
        frameinfo.lineno + 3,
        frameinfo.function,
        block_attrs,
        block_attrs,
        self.test_path,
        frameinfo.lineno + 5,
        frameinfo.function,
        block_attrs,
        block_attrs,
        self.test_path,
        frameinfo.lineno + 6,
        frameinfo.function,
        block_attrs,
        block_attrs,
    )
    actual_log = self._read_text_file(
        params.log_dir_path + '/testLogNestedBlockTextToTextFile.txt'
    )
    self.assertEqual(
        expected_log,
        actual_log,
        'Target code and generated logs are different. Expected log:\n%s\n'
        'Actual log:\n%s\n' % (expected_log, actual_log),
    )

  def testLogNestedBlockTextToCapacitorFile(self):
    # SETUP
    params = sight_pb2.Params()
    params.label = 'testLogNestedBlockTextToCapacitorFile'
    params.log_owner = 'mdb/x-pitchfork'
    params.local = True
    params.capacitor_output = True
    params.log_dir_path = self.create_tempdir().full_path
    # ACT
    frameinfo = inspect.getframeinfo(inspect.currentframe())
    with Sight(params) as sight:
      with Block('outerBlock', sight):
        text('preText', sight)
        with Block('innerBlock', sight):
          text('inText', sight)
        text('postText', sight)
    # ASSERT
    expected_log = [
        sight_pb2.Object(
            location='0000000000',
            index=0,
            attribute=self._create_attributes([], sight),
            file=self.test_path,
            line=frameinfo.lineno + 2,
            func=frameinfo.function,
            ancestor_start_location=['0000000000'],
            sub_type=sight_pb2.Object.ST_BLOCK_START,
            block_start=sight_pb2.BlockStart(label='outerBlock'),
        ),
        sight_pb2.Object(
            location='0000000000:0000000000',
            index=1,
            attribute=self._create_attributes([], sight),
            file=self.test_path,
            line=frameinfo.lineno + 3,
            func=frameinfo.function,
            ancestor_start_location=['0000000000', '0000000000:0000000000'],
            sub_type=sight_pb2.Object.ST_TEXT,
            text=sight_pb2.Text(
                sub_type=sight_pb2.Text.ST_TEXT, text='preText\n'
            ),
        ),
        sight_pb2.Object(
            location='0000000000:0000000001',
            index=2,
            attribute=self._create_attributes([], sight),
            file=self.test_path,
            line=frameinfo.lineno + 4,
            func=frameinfo.function,
            ancestor_start_location=['0000000000', '0000000000:0000000001'],
            sub_type=sight_pb2.Object.ST_BLOCK_START,
            block_start=sight_pb2.BlockStart(label='innerBlock'),
        ),
        sight_pb2.Object(
            location='0000000000:0000000001:0000000000',
            index=3,
            attribute=self._create_attributes([], sight),
            file=self.test_path,
            line=frameinfo.lineno + 5,
            func=frameinfo.function,
            ancestor_start_location=[
                '0000000000',
                '0000000000:0000000001',
                '0000000000:0000000001:0000000000',
            ],
            sub_type=sight_pb2.Object.ST_TEXT,
            text=sight_pb2.Text(
                sub_type=sight_pb2.Text.ST_TEXT, text='inText\n'
            ),
        ),
        sight_pb2.Object(
            location='0000000000:0000000002',
            index=4,
            attribute=self._create_attributes([], sight),
            file=self.test_path,
            line=frameinfo.lineno + 5,
            func=frameinfo.function,
            ancestor_start_location=['0000000000', '0000000000:0000000002'],
            sub_type=sight_pb2.Object.ST_BLOCK_END,
            block_end=sight_pb2.BlockEnd(
                label='innerBlock',
                location_of_block_start='0000000000:0000000001',
                num_direct_contents=1,
                num_transitive_contents=1,
            ),
        ),
        sight_pb2.Object(
            location='0000000000:0000000003',
            index=5,
            attribute=self._create_attributes([], sight),
            file=self.test_path,
            line=frameinfo.lineno + 6,
            func=frameinfo.function,
            ancestor_start_location=['0000000000', '0000000000:0000000003'],
            sub_type=sight_pb2.Object.ST_TEXT,
            text=sight_pb2.Text(
                sub_type=sight_pb2.Text.ST_TEXT, text='postText\n'
            ),
        ),
        sight_pb2.Object(
            location='0000000001',
            index=6,
            attribute=self._create_attributes([], sight),
            file=self.test_path,
            line=frameinfo.lineno + 6,
            func=frameinfo.function,
            ancestor_start_location=['0000000001'],
            sub_type=sight_pb2.Object.ST_BLOCK_END,
            block_end=sight_pb2.BlockEnd(
                label='outerBlock',
                location_of_block_start='0000000000',
                num_direct_contents=3,
                num_transitive_contents=5,
            ),
        ),
    ]
    actual_log = self._read_capacitor_file(
        params.log_dir_path + '/testLogNestedBlockTextToCapacitorFile.capacitor'
    )
    self.assertEqual(len(expected_log), len(actual_log))
    for i in range(0, len(expected_log)):
      compare.assertProtoEqual(
          self,
          expected_log[i],
          actual_log[i],
          'Target code and generated logs are different. Expected'
          ' log[%d]:\n%s\nActual log[%d]:\n%s\n'
          % (i, expected_log[i], i, actual_log[i]),
      )

  def testLogAttributesToTextFile(self):
    # SETUP
    params = sight_pb2.Params()
    params.label = 'testLogAttributesToTextFile'
    params.log_owner = 'mdb/x-pitchfork'
    params.local = True
    params.text_output = True
    params.log_dir_path = self.create_tempdir().full_path
    # ACT
    frameinfo = inspect.getframeinfo(inspect.currentframe())
    with Sight(params) as sight:
      with Attribute('key', 'val', sight):
        text('text', sight)
    # ASSERT
    expected_log = '%s:%d/%s text| %s\n' % (
        self.test_path,
        frameinfo.lineno + 3,
        frameinfo.function,
        self._create_attributes_text(
            [sight_pb2.Attribute(key='key', value='val')], sight
        ),
    )
    actual_log = self._read_text_file(
        params.log_dir_path + '/testLogAttributesToTextFile.txt'
    )
    self.assertEqual(
        expected_log,
        actual_log,
        'Target code and generated logs are different. Expected log:\n%s\n'
        'Actual log:\n%s\n' % (expected_log, actual_log),
    )

  def testLogAttributesToCapacitorFile(self):
    # SETUP
    params = sight_pb2.Params()
    params.label = 'testLogAttributesToCapacitorFile'
    params.log_owner = 'mdb/x-pitchfork'
    params.local = True
    params.capacitor_output = True
    params.log_dir_path = self.create_tempdir().full_path
    # ACT
    frameinfo = inspect.getframeinfo(inspect.currentframe())
    with Sight(params) as sight:
      with Attribute('key', 'val', sight):
        text('text', sight)
    # ASSERT
    expected_log = [
        sight_pb2.Object(
            location='0000000000',
            index=0,
            attribute=self._create_attributes(
                [sight_pb2.Attribute(key='key', value='val')], sight
            ),
            file=self.test_path,
            line=frameinfo.lineno + 3,
            func=frameinfo.function,
            ancestor_start_location=['0000000000'],
            sub_type=sight_pb2.Object.ST_TEXT,
            text=sight_pb2.Text(sub_type=sight_pb2.Text.ST_TEXT, text='text\n'),
        )
    ]
    actual_log = self._read_capacitor_file(
        params.log_dir_path + '/testLogAttributesToCapacitorFile.capacitor'
    )
    self.assertEqual(len(expected_log), len(actual_log))
    for i in range(0, len(expected_log)):
      compare.assertProtoEqual(
          self,
          expected_log[i],
          actual_log[i],
          'Target code and generated logs are different. Expected'
          ' log[%d]:\n%s\nActual log[%d]:\n%s\n'
          % (i, expected_log[i], i, actual_log[i]),
      )


if __name__ == '__main__':
  absltest.main()
