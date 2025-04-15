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
    # Ensure Abseil flags are parsed
    if not FLAGS.is_parsed():
      FLAGS(sys.argv)

    # Default params to be used in Sight
    self.params = sight_pb2.Params(
        label='test-sight',
        bucket_name=f'{os.environ["PROJECT_ID"]}-sight',
    )

  @patch('sight.sight.service.call')
  def test_sight_obj_creation(self, mock_call):
    # Create a mock response for the service call with a dummy ID
    mock_response = MagicMock()
    mock_response.id = 'mock-id-123'
    mock_call.return_value = mock_response  # This will be returned when `service.call()` is used

    # Instantiate the Sight object using test params from self.setUp()
    sight_obj = Sight(self.params)

    # Assert that the object is created successfully
    self.assertIsInstance(sight_obj, Sight)

    # Confirm that the ID returned from the mocked service call is set on the Sight instance
    self.assertEqual(sight_obj.id, 'mock-id-123')

    # Ensure that the service call was made exactly once during initialization
    mock_call.assert_called_once()

  def test_sight_obj_creation_with_silent_logger(self):
    # Enable silent logger mode in params
    self.params.silent_logger = True
    sight_obj = Sight(self.params)
    self.assertIsInstance(sight_obj, Sight)
    # In silent mode, no ID should be generated from a service call — it defaults to 0
    self.assertEqual(sight_obj.id, 0)

  @patch('sight.sight.Sight._close_avro_log')
  @patch('sight.sight.finalize_server')
  def test_close_sight(self, mock_close_avro, mock_finalize_server):
    # Create a Sight object with default params
    sight_obj = Sight(self.params)
    self.assertIsInstance(sight_obj, Sight)

    # Mock text_log and avro_log to simulate file-like resources
    sight_obj.text_log = MagicMock()
    sight_obj.avro_log = MagicMock()
    sight_obj.avro_log.getbuffer().nbytes = 1024  # Ensure avro_log is not empty

    # Call the method under test
    sight_obj.close()

    # Confirm text_log was closed properly
    sight_obj.text_log.close.assert_called_once()
    # Ensure internal Avro cleanup method called properly
    mock_close_avro.assert_called_once()
    # verify finalize_server  mehtod called once
    mock_finalize_server.assert_called_once()


if __name__ == "__main__":
  unittest.main(testRunner=ColorfulTestRunner())
