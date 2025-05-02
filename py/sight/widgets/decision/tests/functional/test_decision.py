"""Tests for the Sight Module."""

import os
from pathlib import Path
import sys
import unittest
from unittest.mock import call
from unittest.mock import MagicMock
from unittest.mock import patch

from absl import flags
from sight import sight
from sight.proto import sight_pb2
from sight.sight import Sight
from sight.widgets.decision import decision
from sight.widgets.decision import decision_episode_fn
from sight.widgets.decision import trials
from sight.widgets.decision.llm_optimizer_client import LLMOptimizerClient
from sight.widgets.decision.single_action_optimizer_client import (
    SingleActionOptimizerClient
)
from sight_service.proto import service_pb2
from tests.colorful_tests import ColorfulTestRunner

FLAGS = flags.FLAGS


class DecisionTest(unittest.TestCase):
  """Tests for the Sight module."""

  def tearDown(self):
    super().tearDown()

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

  @patch.object(decision, 'convert_proto_to_dict')
  def test_get_decision_messages_from_proto(self, mock_convert_proto_to_dict):
    # Create two dummy DecisionMessage protos with specific action_id and action contents
    action1 = sight_pb2.DecisionParam(
        params={
            'region': sight_pb2.Value(string_value="US"),
            'project': sight_pb2.Value(string_value="kokua")
        })
    action2 = sight_pb2.DecisionParam(
        params={
            'region': sight_pb2.Value(string_value="AU"),
            'project': sight_pb2.Value(string_value="kokua-1")
        })

    msg1 = sight_pb2.DecisionMessage(action_id=1, action=action1)
    msg2 = sight_pb2.DecisionMessage(action_id=2, action=action2)

    # Mock the expected return values from convert_proto_to_dict for each message
    mock_convert_proto_to_dict.side_effect = [{
        'region': 'US',
        'project': 'kokua'
    }, {
        'region': 'AU',
        'project': 'kokua-1'
    }]

    # Call the function under test with mocked proto input
    input_proto = [msg1, msg2]
    result = decision.get_decision_messages_from_proto(input_proto)

    # Expected output: dictionary mapping action_id to converted action content
    expected = {
        1: {
            'region': 'US',
            'project': 'kokua'
        },
        2: {
            'region': 'AU',
            'project': 'kokua-1'
        }
    }

    # Validate the function returns the correct mapping
    self.assertEquals(result, expected)

    # Ensure that convert_proto_to_dict was called with the each proto's action
    mock_convert_proto_to_dict.assert_any_call(proto=action1)
    mock_convert_proto_to_dict.assert_any_call(proto=action2)

  def test_execute_run_mode_without_sight_log_id(self):
    # Simulate the absence of sight_log_id flag
    FLAGS.sight_log_id = None

    # Call the function under test and expect it to raise ValueError due to missing flag
    with self.assertRaises(ValueError) as cm:
      decision.execute_run_mode()

    # Ensure that the exception message matches with the expected message
    self.assertEqual(str(cm.exception),
                     'sight_log_id must be provided for decision_mode = run')

  @patch.object(decision.service, 'call')
  @patch('builtins.print')
  def test_execute_run_mode_with_sight_log_id(self, mock_print,
                                              mock_service_call):

    # Set the required flag sight_log_id to enable the run mode
    FLAGS.sight_log_id = '1234'

    # Simulate the expected output from the service call
    mock_service_call.return_value = service_pb2.FetchOptimalActionResponse(
        response_str='ok')

    # Call the function under test
    decision.execute_run_mode()
    mock_service_call.assert_called_once()

    # Verify the function printed expected response
    mock_print.assert_called_with('response:', 'ok')

  @patch.object(sight, 'upload_blob_from_stream')
  @patch.object(sight.service, 'call')
  @patch.object(sight, 'create_external_bq_table')
  #@patch.object(decision, 'execute_local_training')
  @patch.object(decision, 'create_opt_and_start_workers')
  # @patch.object(decision, 'validate_train_mode')
  def test_execute_train_mode(
      self,  #mock_validate_train_mode,
      mock_create_opt_and_start_workers,
      #mock_execute_local_training,
      mock_create_external_bq_table,
      mock_service_call,
      mock_upload_blob):

    FLAGS.worker_mode = 'xyz'
    with self.assertRaises(ValueError) as cm:
      decision.execute_train_mode(None, None, None, None, None, None, None)
    self.assertEqual(str(cm.exception), 'worker mode is not None')

    FLAGS.worker_mode = None
    # Test all supported deployment modes
    distributed_modes = ['cloud_run', 'vm']
    local_modes = ['local']

    # Simulate service.call with different id as response
    mock_service_call.side_effect = [
        service_pb2.CreateResponse(id=1),
        service_pb2.CreateResponse(id=2),
        service_pb2.CreateResponse(id=3),
        service_pb2.CreateResponse(id=4),
        service_pb2.CreateResponse(id=5),
        service_pb2.CreateResponse(id=6)
    ]
    mock_create_external_bq_table.return_value = None

    # Preparing input for the test function
    driver_fn = MagicMock()  # function with user logic
    question_config = {
        "desc":
            "generic question label",
        "attrs_text_proto":
            "py/sight/configs/.text_proto_configs/generic.textproto"
    }
    optimizer_config = {
        "optimizer": "worklist_scheduler",
        "num_questions": 2,
        "mode": "dsub_local_worker",  #"dsub_cloud_worker" #(dsub_local, )
    }
    workers_config = {
        "version": "v0.1",
        "file_path": "py/sight/generic_worker.yaml"
    }
    optimizer_type = "worklist_scheduler"
    question_label = "Q_label1"

    for mode in distributed_modes + local_modes:
      with self.subTest(server_mode=mode):
        # Reset mocks for each subtest to avoid call overlap
        # mock_validate_train_mode.reset_mock()
        mock_create_opt_and_start_workers.reset_mock()
        # mock_execute_local_training.reset_mock()

        FLAGS.server_mode = mode

        # Preparing input for the test function
        sight = Sight(self.params)
        optimizer = decision.Optimizer()
        optimizer.obj = decision.setup_optimizer(sight, optimizer_type)

        decision_configuration = decision.configure_decision(
            sight, question_label, question_config, optimizer_config,
            optimizer.obj)

        # Execute the function under test
        decision.execute_train_mode(sight, decision_configuration, driver_fn,
                                    optimizer_config, workers_config,
                                    optimizer_type, question_label)

        sight.close()

        # Assert correct execution path based on deployment mode
        if FLAGS.server_mode in distributed_modes:
          mock_create_opt_and_start_workers.assert_called_once_with(
              sight, decision_configuration, optimizer_config, workers_config,
              optimizer_type)
        # elif FLAGS.server_mode in local_modes:
        #   mock_execute_local_training.assert_called_once_with(
        #       sight, decision_configuration, driver_fn, optimizer_config,
        #       workers_config, optimizer_type)

    assert mock_service_call.call_count == len(distributed_modes + local_modes)

  @patch.object(sight, 'upload_blob_from_stream')
  @patch.object(sight.service, 'call')
  @patch.object(sight, 'create_external_bq_table')
  @patch.object(trials, 'launch')
  @patch.object(trials, 'start_worker_jobs')
  def test_create_opt_and_start_workers(self, mock_start_worker_jobs,
                                        mock_launch,
                                        mock_create_external_bq_table,
                                        mock_service_call, mock_upload_blob):

    # create parent mock to track the call order of launch and start_worker_jobs
    parent_mock = MagicMock()
    parent_mock.attach_mock(mock_launch, 'launch')
    parent_mock.attach_mock(mock_start_worker_jobs, 'start_worker_jobs')

    sight = Sight(self.params)

    question_config = {
        "desc":
            "generic question label",
        "attrs_text_proto":
            "py/sight/configs/.text_proto_configs/generic.textproto"
    }
    optimizer_config = {
        "optimizer": "worklist_scheduler",
        "num_questions": 2,
        "mode": "dsub_local_worker",  #"dsub_cloud_worker" #(dsub_local, )
    }
    workers_config = {
        "version": "v0.1",
        "file_path": "py/sight/generic_worker.yaml"
    }
    optimizer_type = "worklist_scheduler"
    question_label = "Q_label1"
    optimizer = decision.Optimizer()
    optimizer.obj = decision.setup_optimizer(sight, optimizer_type)

    decision_configuration = decision.configure_decision(
        sight, question_label, question_config, optimizer_config, optimizer.obj)

    # Call the function under test
    decision.create_opt_and_start_workers(sight, decision_configuration,
                                          optimizer_config, workers_config,
                                          optimizer_type)
    sight.close()

    # Assert that each function was called exactly once with correct arguments
    mock_launch.assert_called_once_with(decision_configuration, sight)
    mock_start_worker_jobs.assert_called_once_with(sight, optimizer_config,
                                                   workers_config,
                                                   optimizer_type)

    # verify the order : launch should be called before start_worker_jobs
    expected_order = [
        call.launch(decision_configuration, sight),
        call.start_worker_jobs(sight, optimizer_config, workers_config,
                               optimizer_type)
    ]
    self.assertEqual(parent_mock.mock_calls, expected_order)

  @patch.object(sight, 'upload_blob_from_stream')
  @patch.object(sight.service, 'call')
  @patch.object(sight, 'create_external_bq_table')
  @patch.object(decision, 'finalize_episode')
  @patch.object(decision, 'get_decision_messages_from_proto')
  def test_process_worker_action(self, mock_get_decision_messages,
                                 mock_finalize_episode,
                                 mock_create_external_bq_table,
                                 mock_service_call, mock_upload_blob):
    # Create a mocked response object with dummy decision messages
    response = service_pb2.WorkerAliveResponse(decision_messages=[
        sight_pb2.DecisionMessage(action=sight_pb2.DecisionParam(
            params={
                "a1": sight_pb2.Value(string_value="abc"),
                "a2": sight_pb2.Value(string_value="xyz")
            }))
    ])

    # Simulate parsed decision messages returned by proto converter
    mock_get_decision_messages.return_value = {
        '1': {
            'project': 'kokua'
        },
        '2': {
            'region': 'US'
        }
    }

    sight = Sight(self.params)

    driver_fn = MagicMock()
    env = None
    question_label = "Q_label1"
    optimizer_type = "worklist_scheduler"
    # mock_optimizer.obj.cache = MagicMock()
    optimizer = decision.Optimizer()
    optimizer.obj = decision.setup_optimizer(sight, optimizer_type)
    optimizer.obj.cache = MagicMock()

    # Execute the function under test
    decision.process_worker_action(response, sight, driver_fn, env,
                                   question_label, optimizer.obj)
    sight.close()

    # Ensure proto message parsed once from the reponse
    mock_get_decision_messages.assert_called_once_with(
        decision_messages_proto=response.decision_messages)

    # Ensure finalized episode triggered once at last
    mock_finalize_episode.assert_called_once_with(question_label, sight)


# TODO @Meetatgoogle , resolve this test-case , commenting for now
# def test_get_decision_configuration_for_opt_with_wrong_file_path(self):
#   # Simulate an invalid question_config pointing to a non-existent proto file
#   question_config = {'attrs_text_proto': 'dummy_path/text.proto'}

#   # Expect FileNotFoundError due to missing file
#   with self.assertRaises(FileNotFoundError) as cm:
#     decision.get_decision_configuration_for_opt(None, None, None,
#                                                 question_config, None)

#   # Reconstruct the expected absolute file path based on internal logic
#   current_file = Path(__file__).resolve()
#   sight_repo_path = current_file.parents[6]
#   absoulte_text_proto_path = sight_repo_path.joinpath(
#       question_config['attrs_text_proto'])

#   # Assert the error message matches the path used internally in the function
#   self.assertEqual(str(cm.exception),
#                    f'File not found {absoulte_text_proto_path}')

  @patch.object(sight, 'upload_blob_from_stream')
  @patch.object(sight.service, 'call')
  @patch.object(sight, 'create_external_bq_table')
  @patch.object(decision.os.path, 'exists')
  def test_get_decision_configuration_for_opt(self, mock_path_exist,
                                              mock_create_external_bq_table,
                                              mock_service_call,
                                              mock_upload_blob):
    # Simulate that the proto file exists with valid proto path
    mock_path_exist.return_value = True
    mock_service_call.return_value = service_pb2.CreateResponse(id=123)
    question_config = {
        'attrs_text_proto': 'py/sight/configs/.text_proto_configs/fvs.textproto'
    }

#   # Prepare the input parameters
#   sight = Sight(self.params)
#   question_label = 'sight-test'
#   optimizer_type = "worklist_scheduler"
#   optimizer = decision.Optimizer()
#   optimizer.obj = decision.setup_optimizer(sight, optimizer_type)
#   optimizer_config = {'num_questions': 5}

#   # Call the function under test
#   result = decision.get_decision_configuration_for_opt(
#       sight, question_label, optimizer.obj, question_config, optimizer_config)

#   sight.close()

#   # Assert that a valid DecisionConfigurationStart object is returned
#   self.assertIsInstance(result, sight_pb2.DecisionConfigurationStart)

  @patch.object(sight, 'upload_blob_from_stream')
  @patch.object(sight.service, 'call')
  @patch.object(sight, 'create_external_bq_table')
  def test_setup_optimizer(self, mock_create_external_bq_table,
                           mock_service_call, mock_upload_blob):

    mock_service_call.return_value = service_pb2.CreateResponse(id=123)

    # Prepare inputs for the test function
    sight = Sight(self.params)
    description = ""
    optimizer_type = 'invalid'

    with self.assertRaises(ValueError) as cm:
      result = decision.setup_optimizer(sight, optimizer_type, description)

    # Ensure that the exception message matches with the expected message
    self.assertEqual(str(cm.exception),
                     f'Unknown optimizer type {optimizer_type}')

    optimizer_types = [
        'worklist_scheduler', 'bayesian_opt', 'exhaustive_search', 'vizier',
        'sensitivity_analysis', 'smcpy', 'llm_text_bison_optimize', 'ng_bo'
    ]

    for opt in optimizer_types:
      with self.subTest(optimizer_type=opt):

        # simulate service.call to return response with dummy ID
        mock_service_call.return_value = service_pb2.CreateResponse(id=123)

        # Prepare inputs for the test function
        sight = Sight(self.params)
        description = ""
        optimizer_type = opt

        result = decision.setup_optimizer(sight, optimizer_type, description)
        if optimizer_type.startswith('llm_'):
          self.assertIsInstance(result, LLMOptimizerClient)
        else:
          self.assertIsInstance(result, SingleActionOptimizerClient)

  @patch.object(sight, 'upload_blob_from_stream')
  @patch.object(sight.service, 'call')
  @patch.object(sight, 'create_external_bq_table')
  @patch.object(decision, 'convert_dict_to_proto')
  def test_get_decision_outcome_proto(self, mock_convert_dict_to_proto,
                                      mock_create_external_bq_table,
                                      mock_service_call, mock_upload_blob):
    # simulate service.call to return response with dummy ID
    mock_service_call.return_value = service_pb2.CreateResponse(id=123)

    mock_convert_dict_to_proto.return_value = sight_pb2.DecisionParam(
        params={
            'outcome1': sight_pb2.Value(sub_type=sight_pb2.Value.ST_JSON,
                                        json_value='123')
        })

    # Prepare inputs for the test function
    sight = Sight(self.params)
    outcome_label = "label_1"
    sight.widget_decision_state = {
        "sum_reward": 30.0,
        "discount": 1.0,
        "sum_outcome": {
            'outcome1': '123'
        }
    }

    result = decision.get_decision_outcome_proto(outcome_label, sight)

    expected = sight_pb2.DecisionOutcome(
        outcome_label=outcome_label,
        reward=sight.widget_decision_state['sum_reward'],
        discount=sight.widget_decision_state['discount'],
        outcome_params=sight_pb2.DecisionParam(
            params={
                'outcome1': sight_pb2.Value(
                    sub_type=sight_pb2.Value.ST_JSON,
                    json_value=sight.widget_decision_state['sum_outcome']
                    ['outcome1'])
            }))

    self.assertIsInstance(result, sight_pb2.DecisionOutcome)
    self.assertEqual(result, expected)

  @patch.object(sight, 'upload_blob_from_stream')
  @patch.object(sight, 'create_external_bq_table')
  @patch.object(sight.service, 'call')
  @patch.object(sight.Sight, 'log_object')
  @patch.object(decision, '_update_cached_batch')
  def test_decision_outcome(self, mock_update_cached_batch, mock_log_object,
                            mock_service_call, mock_create_external_bq_table,
                            mock_upload_blob):
    # simulate service.call to return response with dummy ID
    mock_service_call.return_value = service_pb2.CreateResponse(id=123)
    # Prepare inputs for the test function
    sight = Sight(self.params)
    outcome_label = 'label_outcome'
    reward = 12.0
    discount = 1.0
    outcome = {"o1": "o-1", "o2": 2}

    decision.decision_outcome(outcome_label, sight, reward, outcome, discount)
    sight.close()

    self.assertEqual(sight.widget_decision_state['reward'], reward)
    self.assertNotIn('sum_reward',
                     sight.widget_decision_state)  # Should be popped
    self.assertNotIn('sum_outcome',
                     sight.widget_decision_state)  # Should be popped

    mock_log_object.assert_called_once()
    mock_update_cached_batch.assert_called_once_with(sight)

  @patch.object(sight, 'upload_blob_from_stream')
  @patch.object(sight, 'create_external_bq_table')
  @patch.object(sight.service, 'call')
  def test_propose_actions(self, mock_service_call,
                           mock_create_external_bq_table, mock_upload_blob):
    mock_service_call.side_effect = [
        service_pb2.CreateResponse(id=123),  # for Sight creation
        service_pb2.ProposeActionResponse(action_id=777),  # for propose_actions
    ]

    sight = Sight(self.params)
    question_label = "q_label1"
    action_dict = {"a1": 1, "a2": 2}

    result = decision.propose_actions(sight, question_label, action_dict)
    self.assertEqual(result, 777)

if __name__ == "__main__":
  unittest.main(testRunner=ColorfulTestRunner())
