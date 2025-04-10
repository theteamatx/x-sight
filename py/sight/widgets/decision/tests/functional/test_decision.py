"""Tests for the Sight Module."""

import unittest
from unittest.mock import call, patch, MagicMock
from tests.colorful_tests import ColorfulTestRunner
from pathlib import Path
from sight.proto import sight_pb2
from sight.widgets.decision import decision
import os
import sys
from absl import flags
from sight_service.proto import service_pb2
from sight.widgets.decision import trials

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
    msg1.action = {'region': 'US', 'project': 'kokua'}

    msg2 = MagicMock(spec=sight_pb2.DecisionMessage)
    msg2.action_id = '2'
    msg2.action = {'region': 'AU', 'project': 'kokua-1'}

    mock_convert_proto_to_dict.side_effect = [{
        'region': 'US',
        'project': 'kokua'
    }, {
        'region': 'AU',
        'project': 'kokua-1'
    }]

    input_proto = [msg1, msg2]
    result = decision.get_decision_messages_from_proto(input_proto)
    expected = {
        '1': {
            'region': 'US',
            'project': 'kokua'
        },
        '2': {
            'region': 'AU',
            'project': 'kokua-1'
        }
    }
    self.assertEquals(result, expected)

    mock_convert_proto_to_dict.assert_any_call(proto={
        'region': 'US',
        'project': 'kokua'
    })
    mock_convert_proto_to_dict.assert_any_call(proto={
        'region': 'AU',
        'project': 'kokua-1'
    })

  def test_execute_run_mode_without_sight_log_id(self):
    FLAGS.sight_log_id = None
    with self.assertRaises(ValueError) as cm:
      decision.execute_run_mode()

    self.assertEqual(str(cm.exception),
                     'sight_log_id must be provided for decision_mode = run')

  @patch('sight.widgets.decision.decision.service.call')
  @patch('builtins.print')
  def test_execute_run_mode_with_sight_log_id(self, mock_print,
                                              mock_service_call):
    FLAGS.sight_log_id = '1234'
    mock_service_call.return_value = service_pb2.FetchOptimalActionResponse(
        response_str='ok')

    decision.execute_run_mode()
    mock_service_call.assert_called_once()
    mock_print.assert_called_with('response:', 'ok')

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
    sight.widget_decision_state = {'decision_episode_fn': details}

    FLAGS.optimizer_type = 'exhaustive_search'
    FLAGS.num_trials = 12
    with self.assertRaises(ValueError) as cm:
      decision.validate_train_mode(sight)
    self.assertEqual(str(cm.exception),
                     'Max possible value for num_trials is: 9')

    FLAGS.num_trials = 5
    FLAGS.docker_image = None
    with self.assertRaises(ValueError) as cm:
      decision.validate_train_mode(sight)
    self.assertEqual(str(cm.exception),
                     'docker_image must be provided for distributed mode')

  @patch.object(decision, 'execute_local_training')
  @patch.object(decision, 'create_opt_and_start_workers')
  @patch.object(decision, 'validate_train_mode')
  def test_execute_train_mode(self, mock_validate_train_mode,
                              mock_create_opt_and_start_workers,
                              mock_execute_local_training):
    FLAGS.deployment_mode = 'xyz'
    with self.assertRaises(ValueError) as cm:
      decision.execute_train_mode(None, None, None, None, None, None, None)
    self.assertEqual(str(cm.exception), 'Unsupported deployment mode xyz')

    distributed_modes = ['distributed', 'vm']
    local_modes = ['local', 'dsub_local', 'docker_local', 'worker_mode']

    for mode in distributed_modes + local_modes:
      with self.subTest(deployment_mode=mode):

        mock_validate_train_mode.reset_mock()
        mock_create_opt_and_start_workers.reset_mock()
        mock_execute_local_training.reset_mock()

        FLAGS.deployment_mode = mode
        sight = MagicMock()
        decision_configuration = MagicMock()
        driver_fn = MagicMock()
        optimizer_config = MagicMock()
        workers_config = MagicMock()
        optimizer_type = MagicMock()
        question_label = MagicMock()

        decision.execute_train_mode(sight, decision_configuration, driver_fn,
                                    optimizer_config, workers_config,
                                    optimizer_type, question_label)

        if FLAGS.deployment_mode in distributed_modes:
          mock_create_opt_and_start_workers.assert_called_once_with(
              sight, decision_configuration, optimizer_config, workers_config,
              optimizer_type)
        elif FLAGS.deployment_mode in local_modes:
          mock_execute_local_training.assert_called_once_with(
              sight, decision_configuration, driver_fn, optimizer_config,
              workers_config, optimizer_type)

  @patch.object(trials, 'launch')
  @patch.object(trials, 'start_worker_jobs')
  def test_execute_local_training(self, mock_start_worker_jobs, mock_lauch):
    FLAGS.deployment_mode = 'worker_mode'
    decision.execute_local_training(None, None, None, None, None, None)

    sight = MagicMock()
    decision_configuration = MagicMock()
    driver_fn = MagicMock()
    optimizer_config = MagicMock()
    workers_config = MagicMock()
    optimizer_type = MagicMock()

    local_modes = ['local', 'dsub_local', 'docker_local']
    for mode in local_modes:
      with self.subTest(deployment_mode=mode):

        FLAGS.deployment_mode = mode
        mock_lauch.reset_mock()
        mock_start_worker_jobs.reset_mock()

        decision.execute_local_training(sight, decision_configuration,
                                        driver_fn, optimizer_config,
                                        workers_config, optimizer_type)
        mock_lauch.assert_called_once_with(decision_configuration, sight)

        if (FLAGS.deployment_mode == 'dsub_local'):
          mock_start_worker_jobs.assert_called_once_with(
              sight, optimizer_config, workers_config, optimizer_type)
        else:
          mock_start_worker_jobs.assert_not_called()

  @patch.object(trials, 'launch')
  @patch.object(trials, 'start_worker_jobs')
  def test_create_opt_and_start_workers(self, mock_start_worker_jobs,
                                        mock_launch):

    parent_mock = MagicMock()
    parent_mock.attach_mock(mock_launch, 'launch')
    parent_mock.attach_mock(mock_start_worker_jobs, 'start_worker_jobs')

    sight = MagicMock()
    decision_configuration = MagicMock()
    optimizer_config = MagicMock()
    workers_config = MagicMock()
    optimizer_type = MagicMock()

    decision.create_opt_and_start_workers(sight, decision_configuration,
                                          optimizer_config, workers_config,
                                          optimizer_type)

    mock_launch.assert_called_once_with(decision_configuration, sight)
    mock_start_worker_jobs.assert_called_once_with(sight, optimizer_config,
                                                   workers_config,
                                                   optimizer_type)

    # verify the order
    expected_order = [
        call.launch(decision_configuration, sight),
        call.start_worker_jobs(sight, optimizer_config, workers_config,
                               optimizer_type)
    ]
    self.assertEqual(parent_mock.mock_calls, expected_order)

  @patch.object(decision, 'finalize_episode')
  @patch.object(decision, 'get_decision_messages_from_proto')
  @patch.object(decision, 'optimizer')
  def test_process_worker_action(self, mock_optimizer,
                                 mock_get_decision_messages,
                                 mock_finalize_episode):
    response = MagicMock()
    response.decision_messages = ['dm1', 'dm2']

    mock_get_decision_messages.return_value = {
        '1': {
            'project': 'kokua'
        },
        '2': {
            'region': 'US'
        }
    }

    sight = MagicMock()
    driver_fn = MagicMock()
    env = MagicMock()
    question_label = MagicMock()
    mock_optimizer.obj.cache = MagicMock()

    decision.process_worker_action(response, sight, driver_fn, env,
                                   question_label)

    mock_get_decision_messages.assert_called_once_with(
        decision_messages_proto=response.decision_messages)
    mock_finalize_episode.assert_called_once_with(question_label, sight)

  def test_get_decision_configuration_for_opt_with_wrong_file_path(self):
    question_config = {'attrs_text_proto': 'dummy_path/text.proto'}
    with self.assertRaises(FileNotFoundError) as cm:
      decision.get_decision_configuration_for_opt(None, None, None,
                                                  question_config, None)
    current_file = Path(__file__).resolve()
    sight_repo_path = current_file.parents[6]
    absoulte_text_proto_path = sight_repo_path.joinpath(
      question_config['attrs_text_proto'])

    self.assertEqual(str(cm.exception), f'File not found {absoulte_text_proto_path}')

  @patch('sight.widgets.decision.decision.os.path.exists')
  def test_get_decision_configuration_for_opt(self, mock_path_exist):
    mock_path_exist.return_value = True
    question_config = {'attrs_text_proto' :'py/sight/utils/.text_proto_configs/fvs.textproto'}
    sight = MagicMock()
    sight.params.label = 'test_label'
    question_label = 'sight-test'
    opt_obj = MagicMock()
    opt_obj.optimizer_type.return_value = sight_pb2.DecisionConfigurationStart.OptimizerType.OT_BAYESIAN_OPT
    opt_obj.create_config.return_value = sight_pb2.DecisionConfigurationStart.ChoiceConfig()
    optimizer_config = {'num_questions': 5}

    result = decision.get_decision_configuration_for_opt(sight, question_label, opt_obj, question_config,
    optimizer_config)

    self.assertIsInstance(result, sight_pb2.DecisionConfigurationStart)

  


if __name__ == "__main__":
  unittest.main(testRunner=ColorfulTestRunner())
