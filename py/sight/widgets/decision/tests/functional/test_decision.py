"""Tests for the Sight Module."""

import unittest
from unittest.mock import patch, MagicMock
from tests.colorful_tests import ColorfulTestRunner
from sight.proto import sight_pb2
from sight.widgets.decision import decision
import os
import sys
from absl import flags
from sight_service.proto import service_pb2



FLAGS = flags.FLAGS


class DecisionTest(unittest.TestCase):
  """Tests for the Sight module."""

  def tearDown(self):
    super().tearDown()

  def setUp(self):
    super().setUp()
    if not FLAGS.is_parsed():
      FLAGS(sys.argv)

  @patch.object(decision, 'convert_proto_to_dict')
  def test_get_decision_messages_from_proto(self, mock_convert_proto_to_dict):
    msg1 = MagicMock(spec=sight_pb2.DecisionMessage)
    msg1.action_id = '1'
    msg1.action = {'region' : 'US', 'project' : 'kokua'}

    msg2 = MagicMock(spec=sight_pb2.DecisionMessage)
    msg2.action_id = '2'
    msg2.action = {'region' : 'AU', 'project' : 'kokua-1'}

    mock_convert_proto_to_dict.side_effect = [
      {'region' : 'US', 'project' : 'kokua'},
      {'region' : 'AU', 'project' : 'kokua-1'}
    ]

    input_proto = [msg1,msg2]
    result = decision.get_decision_messages_from_proto(input_proto)
    expected = {
      '1': {'region' : 'US', 'project' : 'kokua'},
      '2': {'region' : 'AU', 'project' : 'kokua-1'}
    }
    self.assertEquals(result, expected)

    mock_convert_proto_to_dict.assert_any_call(proto={'region' : 'US', 'project' : 'kokua'})
    mock_convert_proto_to_dict.assert_any_call(proto={'region' : 'AU', 'project' : 'kokua-1'})

  def test_execute_run_mode_without_sight_log_id(self):
    FLAGS.sight_log_id = None
    with self.assertRaises(ValueError) as cm:
      decision.execute_run_mode()

    self.assertEqual(str(cm.exception), 'sight_log_id must be provided for decision_mode = run')

  # @patch.object(decision, 'service.call')
  @patch('sight.widgets.decision.decision.service.call')
  @patch('builtins.print')
  def test_execute_run_mode_with_sight_log_id(self, mock_print, mock_service_call):
    FLAGS.sight_log_id = '1234'
    mock_service_call.return_value = service_pb2.FetchOptimalActionResponse(response_str='ok')

    decision.execute_run_mode()
    mock_service_call.assert_called_once()
    mock_print.assert_called_with('response:','ok')

  # def test_execute_configured_run_mode_without_any_flags(self):
  #   with self.assertRaises(ValueError) as cm:
  #     sight = MagicMock()
  #     driver_fn = MagicMock()
  #     decision.execute_configured_run_mode(sight, driver_fn)
  #   self.assertEqual(str(cm.exception), 'In configured_run mode, decision_run_config_file is required.')

  def test_validate_train_mode(self):
    FLAGS.deployment_mode = 'distributed'
    sight = MagicMock()
    details = MagicMock()

    details.action_max.values.return_value = [10]
    details.action_min.values.return_value = [3]
    sight.widget_decision_state = {'decision_episode_fn' : details}

    FLAGS.optimizer_type = 'exhaustive_search'
    FLAGS.num_trials = 12
    with self.assertRaises(ValueError) as cm:
      decision.validate_train_mode(sight)
    self.assertEqual(str(cm.exception), 'Max possible value for num_trials is: 9')

    FLAGS.num_trials = 5
    FLAGS.docker_image = None
    with self.assertRaises(ValueError) as cm:
      decision.validate_train_mode(sight)
    self.assertEqual(str(cm.exception), 'docker_image must be provided for distributed mode')

if __name__ == "__main__":
  unittest.main(testRunner=ColorfulTestRunner())
