"""Tests for the Sight Module."""

import unittest
from unittest.mock import patch, MagicMock
from tests.colorful_tests import ColorfulTestRunner
from sight.proto import sight_pb2
from sight.sight import Sight
import os
import sys
from absl import flags

FLAGS = flags.FLAGS


class SightTest(unittest.TestCase):
  """Tests for the Sight module."""

  # def tearDown(self):
  #   super().tearDown()

  def setUp(self):
    super().setUp()
    if not FLAGS.is_parsed():
      FLAGS(sys.argv)
    self.params = sight_pb2.Params(
      label='test-sight',
      bucket_name=f'{os.environ["PROJECT_ID"]}-sight',
    )

  @patch('sight.sight.service.call')
  def test_sight_obj_creation(self, mock_call):
    mock_response = MagicMock()
    mock_response.id = 'mock-id-123'
    mock_call.return_value = mock_response

    sight_obj = Sight(self.params)
    self.assertIsInstance(sight_obj, Sight)
    self.assertEqual(sight_obj.id, 'mock-id-123')
    mock_call.assert_called_once()

  def test_sight_obj_creation_with_silent_logger(self):
    self.params.silent_logger = True
    sight_obj = Sight(self.params)
    self.assertIsInstance(sight_obj, Sight)
    self.assertEqual(sight_obj.id, 0)

  @patch('sight.sight.Sight._close_avro_log')
  @patch('sight.sight.finalize_server')
  def test_close_sight(self, mock_close_avro, mock_finalize_server):
    sight_obj = Sight(self.params)
    self.assertIsInstance(sight_obj, Sight)
    sight_obj.text_log = MagicMock()
    sight_obj.avro_log = MagicMock()
    sight_obj.avro_log.getbuffer().nbytes = 1024

    sight_obj.close()
    sight_obj.text_log.close.assert_called_once()
    mock_close_avro.assert_called_once()
    mock_finalize_server.assert_called_once()








if __name__ == "__main__":
  unittest.main(testRunner=ColorfulTestRunner())
