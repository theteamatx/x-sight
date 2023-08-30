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

"""Decisions and their outcomes within the Sight log."""

import inspect
from typing import Any, Callable, Dict, Optional, Text, Tuple
import os
import sys

from absl import flags
from absl import logging
import grpc
import time

from sight.proto import sight_pb2
from service.decision import decision_pb2
from service.decision import decision_pb2_grpc

from sight.widgets.decision.rl import decision_episode_fn
from sight.widgets.decision import trials
from sight.service import generate_metadata

_DECISON_MODE = flags.DEFINE_enum(
    'decision_mode', 'train', ['train', 'run', 'configured_run'],
    ('Indicates whether the decision API should be used to train a decision '
     'model (train) or use it to run the application (run).'))
_DECISION_RUN_CONFIG_FILE = flags.DEFINE_string(
    'decision_run_config_file', None,
    ('File that contains the Sight configuration for driving this '
     'application\'s execution via the Decision API'))
_DECISION_TRAIN_ALG = flags.DEFINE_enum(
    'decision_train_alg', 'vizier_xm',
    ['rl_local', 'rl_xm', 'predict', 'vizier_xm', 'worker_mode', 'vizier_local'],
    ('The procedure to use when training a model to drive applications that '
     'use the Decision API.'))
_OPTIMIZER_TYPE = flags.DEFINE_enum(
    'optimizer_type', 'vizier',
    ['vizier', 'dm_acme'],
    ('The optimizer to use'))
_NUM_TRAIN_WORKERS = flags.DEFINE_integer(
    'num_train_workers', 3, 'Number of workers to use in a training run.')
_DECISION_TRAIN_NUM_EPISODES = flags.DEFINE_integer(
    'decision_train_num_episodes', 0,
    'Number of training episodes to generate for training.')
_DECISION_TRAIN_OUT_CONFIG_FILE = flags.DEFINE_string(
    'decision_train_out_config_file', None,
    ('File to which the configuration that describes the trained model will '
     'be written.'))
_BINARY_PATH = flags.DEFINE_string('binary_path', sys.argv[0], #path of the file that invoked this file 
                                   ('Path of the Blaze target of this binary.'))
# _BINARY_PATH = flags.DEFINE_string('binary_path', 'sight/demo/sweetness.py', #None,
#                                    ('Path of the Blaze target of this binary.'))

_file_name = 'decision.py'

def configure(
    decision_configuration: Optional[sight_pb2.DecisionConfigurationStart],
    widget_decision_state: Dict[str, Any]):
  """Augments the Decision-API specific state within a Sight logger.

  The configuration object contains the state of this widgets and tracks any
  updates to the widget's state within the context of a single Sight
  logger object.

  Args:
    decision_configuration: Proto that describes additional configuration for
      the Decision API.
    widget_decision_state: The object that holds the Decision API's current
      state.

  Returns:
    The dictionary that maps each choice label to the algorithm to be used
    to make the choice.
  """
  method_name = "configure"
  print(f">>>>>>>>>  In {method_name} method of {_file_name} file.")

  if decision_configuration:
    widget_decision_state[
        'choice_algorithm'] = decision_configuration.choice_algorithm

  if 'state' not in widget_decision_state:
    widget_decision_state['state'] = {}

  if 'decision_episode_fn' not in widget_decision_state:
    widget_decision_state['decision_episode_fn'] = None

  if 'rl_decision_driver' not in widget_decision_state:
    widget_decision_state['rl_decision_driver'] = None
  
  print(f"<<<<<<<<<  Out {method_name} method of {_file_name} file.")


def _attr_dict_to_proto(attrs: Dict[str, Tuple[float, float]],
                        attrs_proto: Any):
  """Converts a dict of attribute constraints to its proto representation."""
  for attr_name, attr_range in attrs.items():
    attrs_proto[attr_name].CopyFrom(
        sight_pb2.DecisionConfigurationStart.AttrProps(
            min_value=attr_range[0],
            max_value=attr_range[1],
        ))


def run(
    driver_fn: Callable[[Any], Any],
    state_attrs: Dict[str, Tuple[float, float]],
    action_attrs: Dict[str, Tuple[float, float]],
    sight: Any,
    # binary_path: str
):
  """Driver for running applications that use the Decision API.

  Args:
    driver_fn: Driver function for calling application logic that uses the Sight
      Decision API to describe decisions and their outcomes. It is assumed that
      driver_fn does not maintain state across invocations and can be called as
      many time as needed, possibly concurrently (i.e. does not keep state
      within global variables either internally or via its interactions with
      external resources).
    state_attrs: maps the name of each state variable to its minimum and maximum
      possible values.
    action_attrs: maps the name of each variable that describes possible
      decisions to its minimum and maximum possible values.
    sight: The Sight object to be used for logging.
  """

  method_name = "run"
  print(f">>>>>>>>>  In {method_name} method of {_file_name} file.")

  decision_configuration = sight_pb2.DecisionConfigurationStart()
  _attr_dict_to_proto(state_attrs, decision_configuration.state_attrs)
  _attr_dict_to_proto(action_attrs, decision_configuration.action_attrs)

  # print("decision_configuration : ", decision_configuration)
  
  sight.enter_block(
      'Decision Configuration',
      sight_pb2.Object(
          block_start=sight_pb2.BlockStart(
              sub_type=sight_pb2.BlockStart.ST_CONFIGURATION,
              configuration=sight_pb2.ConfigurationStart(
                  sub_type=sight_pb2.ConfigurationStart
                  .ST_DECISION_CONFIGURATION,
                  decision_configuration=decision_configuration))))
  sight.exit_block('Decision Configuration', sight_pb2.Object())
  sight.widget_decision_state['num_decision_points'] = 0

  sight.widget_decision_state[
      'decision_episode_fn'] = decision_episode_fn.DecisionEpisodeFn(
          driver_fn, state_attrs, action_attrs)

  if _DECISON_MODE.value == 'run':
    print("_DECISON_MODE.value == 'run'")
    driver_fn(sight)
  elif _DECISON_MODE.value == 'configured_run':
    if not _DECISION_RUN_CONFIG_FILE.value:
      raise ValueError(
          'In configured_run mode decision_run_config_file is required.')
    sight.add_config_file(_DECISION_RUN_CONFIG_FILE.value)
    driver_fn(sight)
  elif _DECISON_MODE.value == 'train':
    if _DECISION_TRAIN_ALG.value == 'vizier_xm':
      print("_DECISION_TRAIN_ALG.value == 'vizier_xm'")
      trials.launch(
          _NUM_TRAIN_WORKERS.value,
          _BINARY_PATH.value,
          _OPTIMIZER_TYPE.value,
          sight,
      )
    elif _DECISION_TRAIN_ALG.value == 'worker_mode':
      print("_DECISION_TRAIN_ALG.value == 'worker_mode'")
      for i in range(int(os.environ['num_samples'])):  
        print("*"*30)
        # resetting some Data structure
        if 'constant_action' in sight.widget_decision_state:
          del sight.widget_decision_state['constant_action']
        sight.widget_decision_state['sum_outcome'] = 0
        sight.widget_decision_state['last_reward'] = None
        
        driver_fn(sight)
        finalize_episode(sight)
      

    elif _DECISION_TRAIN_ALG.value == 'predict':
      for _ in range(_DECISION_TRAIN_NUM_EPISODES.value):
        driver_fn(sight)
  print(f"<<<<<<<<<  Out {method_name} method of {_file_name} file.")


def state_updated(
    name: str,
    obj_to_log: Any,
    sight: Any,
) -> None:
  """Called to inform the decision API that the current state has been updated.

  Args:
    name: The name of the updated state variable.
    obj_to_log: The value of the state variable.
    sight: Instance of a Sight logger.
  """
  if sight.widget_decision_state is not None and \
     'decision_episode_fn' in sight.widget_decision_state and \
      sight.widget_decision_state[
      'decision_episode_fn'] and name in sight.widget_decision_state[
          'decision_episode_fn'].state_attrs:
    sight.widget_decision_state['state'][name] = obj_to_log


def decision_point(
    choice_label: str,
    sight: Any,
    # choice: Optional[Callable[[], Dict[Text, float]]],
    # optimizer_type: str
) -> Dict[Text, float]:
  """Documents an execution point when a decision is made.

  If chosen_option is not provided, it is logged into sight. Otherwise, this
  method uses its own decision procedure, guided by the previously observed
  decisions and their outcomes, to make a choice and returns the corresponding
  chosen_option and parameters.

  Args:
    choice_label: Identifies the choice being made.
    sight: Instance of a Sight logger.
    choice: Callable that returns a tuple with (i) The chosen option, and (ii)
      Key-value pairs that that characterize the chosen option.

  Returns:
    Dict that maps the name of each action variable to its chosen value.
  """
  method_name = "decision_point"
  print(f">>>>>>>>>  In {method_name} method of {_file_name} file.")

  sight.widget_decision_state['num_decision_points'] += 1
  chosen_action = None
  
  # actions for vizier already exist.
  if 'constant_action' in sight.widget_decision_state:
    print("not calling server as actions for vizier already exist.")
    return sight.widget_decision_state['constant_action']

  # calling dp method on service directly
  sight_service,metadata = generate_metadata()

  req = decision_pb2.DecisionPointRequest()

  if(_OPTIMIZER_TYPE.value == "vizier"):
    req.optimizer_type = decision_pb2.OptimizerType.OT_VIZIER
    req.client_id = os.environ['PARENT_LOG_ID']
    req.worker_id = "client_"+os.environ['PARENT_LOG_ID']+"_worker_"+os.environ['worker_location']
    # req.vizier_study_name = os.environ['VIZIER_STUDY_NAME']
  elif(_OPTIMIZER_TYPE.value == "dm_acme"):
    req.optimizer_type = decision_pb2.OptimizerType.OT_ACME
    decision_point = sight_pb2.DecisionPoint()

    # sorting the dictionary so, in each call it should be consistent
    state_attrs_keys = list(sight.widget_decision_state["state"].keys())
    state_attrs_keys.sort()
    sorted_state_attrs_dict = {}
    for k in state_attrs_keys : 
      sorted_state_attrs_dict[k] = sight.widget_decision_state["state"][k]
    
    for (k,v) in sorted_state_attrs_dict.items():
      decision_point.state_params[k] = v
    req.decision_point.CopyFrom(decision_point)
  else:
    req.optimizer_type = decision_pb2.OptimizerType.OT_UNKNOWN


  backoff_interval = 1
  while True:
    try:
      # print("decision point req is : ", req)
      response = sight_service.DecisionPointMethod(req,  300, metadata=metadata)
      logging.info('##### response=%s #####', response.actions)
      break
    except grpc.RpcError as e:
      if e.code() == grpc.StatusCode.DEADLINE_EXCEEDED or "DEADLINE_EXCEEDED" in e.details():
          print("deadline exceeded one time, trying again...")
          time.sleep(backoff_interval)
          print("backed off for "+str(backoff_interval)+" seconds...")
          backoff_interval *= 2
          continue
      else:
          logging.info('RPC ERROR: %s', e)
      

  received_actions = response.actions
  print("received_actions : ", received_actions)
  
  chosen_action = {}
  for key, value in received_actions.items():
    chosen_action[key] = value

  # choosen_action will be same for all calls of decision point in vizier, caching it here
  if _OPTIMIZER_TYPE.value == "vizier":
    sight.widget_decision_state['constant_action'] = chosen_action

  choice_params = {}
  for attr in sight.widget_decision_state['decision_episode_fn'].action_attrs:
    if attr in chosen_action:
      choice_params[attr] = str(chosen_action[attr])
  # print("choice_label : ", choice_label)
  # print("choice_params : ", choice_params)

  # pytype: disable=attribute-error
  sight.log_object(
      sight_pb2.Object(
          sub_type=sight_pb2.Object.ST_DECISION_POINT,
          decision_point=sight_pb2.DecisionPoint(
              choice_label=choice_label,
              choice_params=choice_params,
          )),
      inspect.currentframe().f_back.f_back)
  # pytype: enable=attribute-error

  # print("chosen_action : ",chosen_action)
  print(f"<<<<<<<<<  Out {method_name} method of {_file_name} file.")
  return chosen_action


def decision_outcome(
    outcome_label: str,
    outcome_value: float,
    sight: Any,
    # optimizer_type: str
) -> None:
  """Documents a the outcome of prior decisions.

  Args:
    outcome_label: Label that identifies the outcome.
    outcome_value: The numeric value of this outcome, with higher values being
      more desirable.
    sight: Instance of a Sight logger.
  """
  method_name = "decision_outcome"
  print(f">>>>>>>>>  In {method_name} method of {_file_name} file.")

  sight.widget_decision_state['last_reward'] = outcome_value
  print("last_reward : ", outcome_value)

  # if 'sum_outcome' not in sight.widget_decision_state:
  #   sight.widget_decision_state['sum_outcome'] = 0
  sight.widget_decision_state['sum_outcome'] += outcome_value
  print("sum_outcome : ", sight.widget_decision_state['sum_outcome'])

  # In case of vizier (no constant action), no call to service
  if 'constant_action' not in sight.widget_decision_state:
    sight_service,metadata = generate_metadata()

    req = decision_pb2.DecisionOutcomeRequest()
    req.last_call = False

    if(_OPTIMIZER_TYPE.value == "vizier"):
      # will never be called for vizier
      pass
    elif(_OPTIMIZER_TYPE.value == "dm_acme"):
      req.optimizer_type = decision_pb2.OptimizerType.OT_ACME
      req.last_reward = sight.widget_decision_state['last_reward']
    else:
      req.optimizer_type = decision_pb2.OptimizerType.OT_UNKNOWN
    try:
      response = sight_service.DecisionOutcomeMethod(req,  300, metadata=metadata)
      logging.info('##### response=%s #####', response.response_str)
    except Exception as e:
      logging.info('RPC ERROR: %s', e)
    
    if(response.response_str == "Success!"):
      print("Observe completed")
    else:
      print("!!ERROR!!")

  sight.log_object(
      sight_pb2.Object(
          sub_type=sight_pb2.Object.ST_DECISION_OUTCOME,
          decision_outcome=sight_pb2.DecisionOutcome(
              outcome_label=outcome_label, outcome_value=outcome_value)),
      inspect.currentframe().f_back.f_back)

  print(f"<<<<<<<<<  Out {method_name} method of {_file_name} file.")
  
def finalize_episode(sight):
  method_name = "finalize_episode"
  print(f">>>>>>>>>  In {method_name} method of {_file_name} file.")
  
  if _DECISION_TRAIN_ALG.value == 'worker_mode':
    sight_service,metadata = generate_metadata()

    req = decision_pb2.DecisionOutcomeRequest()
    req.last_call = True

    if(_OPTIMIZER_TYPE.value == "vizier"):
      req.optimizer_type = decision_pb2.OptimizerType.OT_VIZIER
      req.client_id = os.environ['PARENT_LOG_ID']
      req.worker_id = "client_"+os.environ['PARENT_LOG_ID']+"_worker_"+os.environ['worker_location']
      decision_outcome = sight_pb2.DecisionOutcome()
      decision_outcome.outcome_label = 'outcome'
      decision_outcome.outcome_value = sight.widget_decision_state['sum_outcome']
      req.decision_outcome.CopyFrom(decision_outcome)    
    elif(_OPTIMIZER_TYPE.value == "dm_acme"):
      req.optimizer_type = decision_pb2.OptimizerType.OT_ACME
      req.last_reward = sight.widget_decision_state['last_reward']
    else:
      req.optimizer_type = decision_pb2.OptimizerType.OT_UNKNOWN
    
    try:
      # print("decision outcome req here is : ", req)
      response = sight_service.DecisionOutcomeMethod(req,  300, metadata=metadata)
      logging.info('##### response=%s #####', response.response_str)
    except Exception as e:
      logging.info('RPC ERROR: %s', e)

    if(response.response_str == "Success!"):
      print("episode completed....")
    else:
      print("!!ERROR!! : calling finalize_episode in decision")
  else:
    print('Not in worker mode, so skipping it')

  print(f"<<<<<<<<<  Out {method_name} method of {_file_name} file.")
