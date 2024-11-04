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
import json
import os
import sys
import threading
import time
from typing import Any, Callable, Dict, List, Optional, Text

from absl import flags
# import dm_env
# from absl import logging
from helpers.logs.logs_handler import logger as logging
import numpy as np
import pandas as pd
from sight import service_utils as service
from sight.proto import sight_pb2
from sight.utility import poll_network_batch_outcome
# from sight.widgets.decision.cartpole_driver import driver_fn
from sight.widgets.decision import decision_episode_fn
from sight.widgets.decision import trials
from sight.widgets.decision import utils
# from sight.widgets.decision.acme.acme_optimizer_client import (
#     AcmeOptimizerClient
# )
from sight.widgets.decision.env_driver import driver_fn
from sight.widgets.decision.llm_optimizer_client import LLMOptimizerClient
from sight.widgets.decision.single_action_optimizer_client import (
    SingleActionOptimizerClient
)
from sight_service.proto import service_pb2

# logging.basicConfig(level=logging.DEBUG)

_DECISON_MODE = flags.DEFINE_enum(
    'decision_mode',
    None,
    ['train', 'run', 'configured_run'],
    ('Indicates whether the decision API should be used to train a decision '
     'model (train) or use it to run the application (run).'),
)
_DECISION_RUN_CONFIG_FILE = flags.DEFINE_string(
    'decision_run_config_file',
    None,
    ('File that contains the Sight configuration for driving this '
     "application's execution via the Decision API"),
)
_DECISION_PARAMS = flags.DEFINE_string(
    'decision_params',
    None,
    ('Assignment of the action attributes for the decision. Format: ' +
     '"key1=val1:key2=val2:...".'),
)
_OPTIMIZER_TYPE = flags.DEFINE_enum(
    'optimizer_type',
    None,
    [
        'vizier', 'dm_acme', 'genetic_algorithm', 'exhaustive_search',
        'llm_text_bison_optimize', 'llm_chat_bison_optimize',
        'llm_gemini_pro_optimize', 'llm_text_bison_recommend',
        'llm_chat_bison_recommend', 'llm_gemini_pro_recommend',
        'llm_text_bison_interactive', 'llm_chat_bison_interactive',
        'llm_gemini_pro_interactive', 'bayesian_opt', 'sensitivity_analysis',
        'ng_auto', 'ng_bo', 'ng_cma', 'ng_two_points_de', 'ng_random_search',
        'ng_pso', 'ng_scr_hammersley_search', 'ng_de', 'ng_cga', 'ng_es',
        'ng_dl_opo', 'ng_dde', 'ng_nmm', 'ng_tiny_spsa', 'ng_voronoi_de',
        'ng_cma_small', 'smcpy', 'worklist_scheduler'
    ],
    'The optimizer to use',
)
_NUM_TRAIN_WORKERS = flags.DEFINE_integer(
    'num_train_workers', 1, 'Number of workers to use in a training run.')
_NUM_TRIALS = flags.DEFINE_integer('num_trials', None,
                                   'Number of trials to perform.')
_DECISION_TRAIN_OUT_CONFIG_FILE = flags.DEFINE_string(
    'decision_train_out_config_file',
    None,
    ('File to which the configuration that describes the trained model will '
     'be written.'),
)
_BINARY_PATH = flags.DEFINE_string(
    'binary_path',
    sys.argv[0],  # path of the file that invoked this file
    'Path of the Blaze target of this binary.',
)
_SERVICE_ACCOUNT = flags.DEFINE_string(
    'service_account',
    # None,
    'sight-service-account',
    'service account to call sight-service',
)
_GCLOUD_DIR_PATH = flags.DEFINE_string(
    'gcloud_dir_path',
    f'{os.path.expanduser("~")}/.config/gcloud',
    # None,#'/usr/local/google/home/meetashah/.config/gcloud',
    'path of gcloud dir in local system',
)
_DOCKER_IMAGE = flags.DEFINE_string(
    'docker_image',
    None,  #'gcr.io/cameltrain/sight-worker-meet',
    'image path in gcr.io for worker code',
)
_WORKER_MODE = flags.DEFINE_enum(
    'worker_mode',
    None,
    ['dsub_local_worker', 'dsub_cloud_worker', 'docker_local_worker'],
    'Mode of workers to be spawned via dsub',
)

_DISTRIBUTED_ACTOR = flags.DEFINE_bool(
    'distributed_actor',
    True,
    'running actor at client side and learner at server side',
)

_ENV_NAME = flags.DEFINE_enum(
    'env_name',
    None,
    ['CartPole-v1', 'MountainCar-v0', 'Pendulum-v1', 'ALE/Pong-v5', 'None'],
    'What environment to run',
)

_TRAINED_MODEL_LOG_ID = flags.DEFINE_string(
    'sight_log_id', None, 'Sight log Id of trained run to be used')

_file_name = 'decision_actor.py'
_sight_id = None
_rewards = []
FLAGS = flags.FLAGS


def configure(
    decision_configuration: Optional[sight_pb2.DecisionConfigurationStart],
    widget_decision_state: Dict[str, Any],
):
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
  method_name = 'configure'
  logging.debug('>>>>>>>>>  In %s of %s', method_name, _file_name)

  if decision_configuration:
    widget_decision_state['choice_config'] = (
        decision_configuration.choice_config)

  if 'state' not in widget_decision_state:
    widget_decision_state['state'] = {}

  if 'decision_episode_fn' not in widget_decision_state:
    widget_decision_state['decision_episode_fn'] = None

  if 'rl_decision_driver' not in widget_decision_state:
    widget_decision_state['rl_decision_driver'] = None

  logging.debug("<<<<  Out %s of %s", method_name, _file_name)


def init_sight_polling_thread(sight_id):
  # print
  status_update_thread = threading.Thread(target=poll_network_batch_outcome,
                                          args=(sight_id,))
  print('*************** starting thread ************')
  status_update_thread.start()


def attr_dict_to_proto(
    attrs: Dict[str, sight_pb2.DecisionConfigurationStart.AttrProps],
    attrs_proto: Any,
):
  """Converts a dict of attribute constraints to its proto representation."""
  for attr_name, attr_details in attrs.items():
    attrs_proto[attr_name].CopyFrom(attr_details)


class Optimizer:

  def __init__(self):
    self.obj = None

  def get_instance(self):
    return self.obj


optimizer = Optimizer()


def attr_to_dict(attr, array):
  """Converts a spec type array to a dict of attribute constraints.

  Args:
    array: The spec array to be converted.
    attr: The name of the attribute.

  Returns:
    A dict of attribute constraints.
  """
  result = {}
  method_name = 'attr_to_dict'
  logging.debug('>>>>>>>>>  In %s of %s', method_name, _file_name)
  # print('Array : ', array)
  # if(array.dtype == np.float32):
  #   dtype = sight_pb2.DecisionConfigurationStart.DataType.DT_FLOAT32
  # elif(array.dtype == np.int64):
  #   dtype = sight_pb2.DecisionConfigurationStart.DataType.DT_INT64

  # default
  # dtype = sight_pb2.DecisionConfigurationStart.DataType.DT_FLOAT32

  if isinstance(array, dm_env.specs.DiscreteArray):
    valid_values = []
    for i in range(array.num_values):
      valid_values.append(i)
    if array.shape == ():
      key = f'{attr}_{1}'
      result[key] = sight_pb2.DecisionConfigurationStart.AttrProps(
          valid_int_values=valid_values)

  elif isinstance(array, dm_env.specs.BoundedArray):
    if array.shape == () or array.shape == (1,):
      # minimum = float(array.minimum if array.minimum.size == 1 else array.minimum[0])
      # maximum = float(array.maximum if array.maximum.size == 1 else array.maximum[0])
      minimum = float(array.minimum[0])
      maximum = float(array.maximum[0])
      key = f'{attr}_{1}'
      result[key] = sight_pb2.DecisionConfigurationStart.AttrProps(
          min_value=minimum,
          max_value=maximum,
          # datatype=dtype
      )
    else:
      minimum = np.repeat(
          array.minimum,
          array.shape[0]) if array.minimum.size == 1 else array.minimum
      maximum = np.repeat(
          array.maximum,
          array.shape[0]) if array.maximum.size == 1 else array.maximum

      for i in range(array.shape[0]):
        key = f'{attr}_{i + 1}'
        result[key] = sight_pb2.DecisionConfigurationStart.AttrProps(
            min_value=float(minimum[i]),
            max_value=float(maximum[i]),
            # datatype=dtype
        )
  # todo : need to handle this case when specs are in different form
  else:
    for i in range(array.shape[0]):
      key = f'{attr}_{i + 1}'
      result[key] = sight_pb2.DecisionConfigurationStart.AttrProps()

  logging.debug("<<<<  Out %s of %s", method_name, _file_name)
  return result


def run(
    sight: Any,
    env: Any = None,
    driver_fn: Callable[[Any], Any] = driver_fn,
    state_attrs: Dict[str, sight_pb2.DecisionConfigurationStart.AttrProps] = {},
    action_attrs: Dict[str,
                       sight_pb2.DecisionConfigurationStart.AttrProps] = {},
    outcome_attrs: Dict[str,
                        sight_pb2.DecisionConfigurationStart.AttrProps] = {},
    description: str = '',
):
  """Driver for running applications that use the Decision API.

  Args:
    sight: The Sight object to be used for logging.
    env: environment object if passed by user.
    driver_fn: Driver function for calling application logic that uses the Sight
      Decision API to describe decisions and their outcomes. It is assumed that
      driver_fn does not maintain state across invocations and can be called as
      many time as needed, possibly concurrently (i.e. does not keep state
      within global variables either internally or via its interactions with
      external resources).
    state_attrs: Maps the name of each state variable to its possible values.
    action_attrs: Maps the name of each variable that describes possible
      decisions to its possible values.
    outcome_attrs: Maps the name of each variable that describes possible
      outcomes to its possible values.
    description: Human-readable description of the application.
  """

  method_name = 'run'
  logging.debug('>>>>>>>>>  In %s of %s', method_name, _file_name)

  if env is not None:
    if state_attrs == {}:
      state_attrs = attr_to_dict(env.observation_spec(), 'state')
    if action_attrs == {}:
      action_attrs = attr_to_dict(env.action_spec(), 'action')

  sight.widget_decision_state['decision_episode_fn'] = (
      decision_episode_fn.DecisionEpisodeFn(driver_fn, state_attrs,
                                            action_attrs))
  # print(sight.widget_decision_state['decision_episode_fn'])
  # raise SystemError

  if _OPTIMIZER_TYPE.value == 'dm_acme':
    optimizer.obj = AcmeOptimizerClient(sight)
  elif _OPTIMIZER_TYPE.value == 'vizier':
    optimizer.obj = SingleActionOptimizerClient(
        sight_pb2.DecisionConfigurationStart.OptimizerType.OT_VIZIER, sight)
  elif _OPTIMIZER_TYPE.value == 'genetic_algorithm':
    optimizer.obj = GeneticAlgorithmOptimizerClient(
        max_population_size=_NUM_TRAIN_WORKERS.value, sight=sight)
  elif _OPTIMIZER_TYPE.value == 'exhaustive_search':
    optimizer.obj = SingleActionOptimizerClient(
        sight_pb2.DecisionConfigurationStart.OptimizerType.OT_EXHAUSTIVE_SEARCH,
        sight)
  elif _OPTIMIZER_TYPE.value.startswith('llm_'):
    optimizer.obj = LLMOptimizerClient(
        _OPTIMIZER_TYPE.value.partition('llm_')[2], description, sight)
  elif _OPTIMIZER_TYPE.value == 'bayesian_opt':
    optimizer.obj = SingleActionOptimizerClient(
        sight_pb2.DecisionConfigurationStart.OptimizerType.OT_BAYESIAN_OPT,
        sight)
  elif _OPTIMIZER_TYPE.value == 'sensitivity_analysis':
    optimizer.obj = SingleActionOptimizerClient(
        sight_pb2.DecisionConfigurationStart.OptimizerType.
        OT_SENSITIVITY_ANALYSIS, sight)
  elif _OPTIMIZER_TYPE.value.startswith('ng_'):
    optimizer.obj = SingleActionOptimizerClient(
        sight_pb2.DecisionConfigurationStart.OptimizerType.OT_NEVER_GRAD, sight,
        _OPTIMIZER_TYPE.value.partition('ng_')[2])
  elif _OPTIMIZER_TYPE.value == 'smcpy':
    optimizer.obj = SingleActionOptimizerClient(
        sight_pb2.DecisionConfigurationStart.OptimizerType.OT_SMC_PY, sight)
  elif _OPTIMIZER_TYPE.value == 'worklist_scheduler':
    optimizer.obj = SingleActionOptimizerClient(
        sight_pb2.DecisionConfigurationStart.OptimizerType.
        OT_WORKLIST_SCHEDULER, sight)
  else:
    raise ValueError(f'Unknown optimizer type {_OPTIMIZER_TYPE.value}')

  if env is not None:
    if state_attrs == {}:
      state_attrs = attr_to_dict(env.observation_spec(), 'state')
    if action_attrs == {}:
      action_attrs = attr_to_dict(env.action_spec(), 'action')
  if outcome_attrs == {}:
    outcome_attrs = {
        'outcome': sight_pb2.DecisionConfigurationStart.AttrProps()
    }

  decision_configuration = sight_pb2.DecisionConfigurationStart()
  decision_configuration.optimizer_type = optimizer.obj.optimizer_type()

  if (_NUM_TRIALS.value):
    decision_configuration.num_trials = _NUM_TRIALS.value
  # if FLAGS.deployment_mode == 'worker_mode':
  #     decision_configuration.num_trials = int(os.environ['num_samples'])
  # else:
  #     decision_configuration.num_trials = _NUM_TRIALS.value
  decision_configuration.choice_config[sight.params.label].CopyFrom(
      optimizer.obj.create_config())
  attr_dict_to_proto(state_attrs, decision_configuration.state_attrs)
  attr_dict_to_proto(action_attrs, decision_configuration.action_attrs)
  attr_dict_to_proto(outcome_attrs, decision_configuration.outcome_attrs)

  sight.enter_block(
      'Decision Configuration',
      sight_pb2.Object(block_start=sight_pb2.BlockStart(
          sub_type=sight_pb2.BlockStart.ST_CONFIGURATION,
          configuration=sight_pb2.ConfigurationStart(
              sub_type=sight_pb2.ConfigurationStart.ST_DECISION_CONFIGURATION,
              decision_configuration=decision_configuration,
          ),
      )),
  )
  sight.exit_block('Decision Configuration', sight_pb2.Object())
  sight.widget_decision_state['num_decision_points'] = 0

  sight.widget_decision_state['decision_episode_fn'] = (
      decision_episode_fn.DecisionEpisodeFn(driver_fn, state_attrs,
                                            action_attrs))
  sight.widget_decision_state['proposed_actions'] = []

  if _DECISON_MODE.value == 'run':
    logging.info('_DECISON_MODE.value == run')
    # sight.widget_decision_state['sum_outcome'] = 0
    # sight.widget_decision_state['last_reward'] = None
    # if env:
    #   driver_fn(env, sight)
    # else:
    #   driver_fn(sight)
    # finalize_episode(sight)

    if (not FLAGS.sight_log_id):
      raise ValueError(
          "sight_log_id have to be passed from the trained run for decision_mokde = run"
      )

    req = service_pb2.FetchOptimalActionRequest(
        client_id=FLAGS.sight_log_id,
        # worker_id=f'client_{client_id}_worker_{worker_location}',
    )
    response = service.call(
        lambda s, meta: s.FetchOptimalAction(req, 300, metadata=meta))
    print('response : ', response.response_str)

  elif _DECISON_MODE.value == 'configured_run':
    # ? not proper flow right now
    # If the run configuration is provided in a file.
    # if _DECISION_RUN_CONFIG_FILE.value:
    if flags.FLAGS.decision_run_config_file:
      sight.add_config_file(_DECISION_RUN_CONFIG_FILE.value)
    # If the run configuration is provided on the command line.
    elif _DECISION_PARAMS.value:
      chosen_action = {}
      for key_val in _DECISION_PARAMS.value.split(':'):
        key, val = tuple(key_val.split('='))
        chosen_action[key] = float(val)
      sight.widget_decision_state['constant_action'] = chosen_action
      # sight.widget_decision_state['sum_outcome'] = 0
      sight.widget_decision_state['last_reward'] = None
    else:
      raise ValueError(
          'In configured_run mode decision_run_config_file is required.')

    # If a docker image is provided, run within it.
    logging.info(
        'decision_train_alg=%s docker_image=%s',
        FLAGS.deployment_mode,
        _DOCKER_IMAGE.value,
    )
    if FLAGS.deployment_mode == 'local' and _DOCKER_IMAGE.value:
      trials.start_job_in_docker(
          1,
          _BINARY_PATH.value,
          _OPTIMIZER_TYPE.value,
          _DOCKER_IMAGE.value,
          _DECISON_MODE.value,
          'docker_worker',
          'worker_mode',
          _DECISION_PARAMS.value,
          sight,
      )
    # Otherwise, run within the current process.
    else:
      driver_fn(sight)

  elif _DECISON_MODE.value == 'train':
    details = sight.widget_decision_state['decision_episode_fn']
    possible_actions = list(details.action_max.values())[0] - list(
        details.action_min.values())[0] + 2

    print('_DECISON_MODE.value : ', _DECISON_MODE.value)
    if FLAGS.deployment_mode in ['distributed', 'vm']:
      if (_OPTIMIZER_TYPE.value == 'exhaustive_search' and
          possible_actions < _NUM_TRIALS.value):
        raise ValueError(
            f"max possible value for num_trials is : {possible_actions}")
      # logging.info('FLAGS.deployment_mode == distributed')
      if (not _DOCKER_IMAGE.value):
        raise ValueError("docker_image must be provided for distributed mode")
      # print("decision_config : ", decision_configuration)
      trials.launch(
          optimizer.obj,
          decision_configuration,
          _NUM_TRAIN_WORKERS.value,
          sight,
      )
      trials.start_jobs(
          _NUM_TRAIN_WORKERS.value,
          _BINARY_PATH.value,
          _OPTIMIZER_TYPE.value,
          _DOCKER_IMAGE.value,
          _DECISON_MODE.value,
          'worker_mode',
          'dsub_cloud_worker',
          sight,
      )
    elif FLAGS.deployment_mode in [
        'local',
        'dsub_local',
        'docker_local',
        'worker_mode',
    ]:
      if FLAGS.deployment_mode == 'worker_mode' or 'PARENT_LOG_ID' in os.environ:
        # not used anymore - for worklist scheduler
        # num_samples_to_run = int(os.environ['num_samples'])
        pass
      else:
        trials.launch(
            optimizer.obj,
            decision_configuration,
            _NUM_TRAIN_WORKERS.value,
            sight,
        )
        # not used anymore - for worklist scheduler
        num_samples_to_run = _NUM_TRIALS.value

      # If a docker image is provided, run within it.
      if (FLAGS.deployment_mode == 'docker_local'
         ):  # and _NUM_TRAIN_WORKERS.value==1:
        trials.start_job_in_docker(
            _NUM_TRIALS.value,
            _BINARY_PATH.value,
            _OPTIMIZER_TYPE.value,
            _DOCKER_IMAGE.value,
            _DECISON_MODE.value,
            'worker_mode',
            'docker_local_worker',
            _DECISION_PARAMS.value,
            sight,
        )
      # run d-sub locally
      elif (FLAGS.deployment_mode == 'dsub_local'
           ):  # and _NUM_TRAIN_WORKERS.value>1:
        trials.start_job_in_dsub_local(
            _NUM_TRAIN_WORKERS.value,
            _NUM_TRIALS.value,
            _BINARY_PATH.value,
            _OPTIMIZER_TYPE.value,
            _DOCKER_IMAGE.value,
            _DECISON_MODE.value,
            'worker_mode',
            'dsub_local_worker',
            sight,
        )
      # Otherwise, run within the current process.
      else:  # local & worker_mode
        # if _OPTIMIZER_TYPE.value == 'dm_acme':
        #   optimizer.obj = acme_optimizer_client.Acme(sight)
        # elif _OPTIMIZER_TYPE.value == 'vizier':
        #   optimizer.obj = vizier_optimizer_client.Vizier(sight)
        # elif _OPTIMIZER_TYPE.value == 'exhaustive_search':
        #   optimizer.obj = exhaustive_search_client.ExhaustiveSearch(sight)

        # actions_list = [
        #                 {'action_1': 1, 'action_2': 1, 'action_3': 1},
        #                 {'action_1': 2, 'action_2': 2, 'action_3': 2},
        #                 {'action_1': 3, 'action_2': 3, 'action_3': 3}
        #             ]
        # unique_action_ids = propose_actions(sight, actions_list)

        if FLAGS.deployment_mode == 'local':
          client_id = str(sight.id)
          worker_location = '0'
        elif (FLAGS.deployment_mode == 'worker_mode'
              # or FLAGS.deployment_mode == 'docker_mode'
             ):
          client_id = os.environ['PARENT_LOG_ID']
          worker_location = os.environ['worker_location']

      # for _ in range(num_samples_to_run):
      # if(FLAGS.optimizer_type == "worklist_scheduler"):
      # if (FLAGS.deployment_mode == 'worker_mode'):
        while (True):
          # #? new rpc just to check move forward or not?
          req = service_pb2.WorkerAliveRequest(
              client_id=client_id,
              worker_id=f'client_{client_id}_worker_{worker_location}')
          response = service.call(
              lambda s, meta: s.WorkerAlive(req, 300, metadata=meta))

          logging.info("response from workAlive rpc is : %s",
                       response.status_type)
          if (response.status_type ==
              service_pb2.WorkerAliveResponse.StatusType.ST_DONE):
            break
          elif (response.status_type ==
                service_pb2.WorkerAliveResponse.StatusType.ST_RETRY):
            logging.info('sleeping for 5 seconds......')
            time.sleep(5)
          elif (response.status_type ==
                service_pb2.WorkerAliveResponse.StatusType.ST_ACT):
            sight.enter_block('Decision Sample', sight_pb2.Object())
            if 'constant_action' in sight.widget_decision_state:
              del sight.widget_decision_state['constant_action']
            sight.widget_decision_state['discount'] = 0
            sight.widget_decision_state['last_reward'] = None

            if env:
              driver_fn(env, sight)
            else:
              driver_fn(sight)

            finalize_episode(sight)
            sight.exit_block('Decision Sample', sight_pb2.Object())
          else:
            raise ValueError("invalid response from server")
        logging.info('exiting from the loop.....')
      # else:
      #   for _ in range(num_samples_to_run):
      #     sight.enter_block('Decision Sample', sight_pb2.Object())
      #     if 'constant_action' in sight.widget_decision_state:
      #         del sight.widget_decision_state['constant_action']
      #     sight.widget_decision_state['discount'] = 0
      #     sight.widget_decision_state['last_reward'] = None

      #     if env:
      #         driver_fn(env, sight)
      #     else:
      #         driver_fn(sight)
      #     finalize_episode(sight)
      #     sight.exit_block('Decision Sample', sight_pb2.Object())

      # req = service_pb2.TestRequest(client_id=str(sight.id))
      # response = service.call(
      #     lambda s, meta: s.PrintInsertionTime(req, 300, metadata=meta)
      # )

    logging.debug("<<<<  Out %s of %s", method_name, _file_name)



def get_state_attrs(sight: Any) -> list[str]:
  state_attrs = []
  state_details = sight.widget_decision_state['decision_episode_fn']

  for i in range(len(state_details.state_attrs)):
    state_attrs.append(state_details.state_attrs[i])
  return state_attrs


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
  if (sight.widget_decision_state is not None and
      'decision_episode_fn' in sight.widget_decision_state and
      sight.widget_decision_state['decision_episode_fn'] and
      name in sight.widget_decision_state['decision_episode_fn'].state_attrs):
    sight.widget_decision_state['state'][name] = obj_to_log


def get_decision_outcome_proto(outcome_label: str,
                               sight: Any) -> sight_pb2.DecisionOutcome:
  decision_outcome = sight_pb2.DecisionOutcome(outcome_label=outcome_label)
  if 'sum_reward' in sight.widget_decision_state:
    decision_outcome.reward = sight.widget_decision_state['sum_reward']

  if 'sum_outcome' in sight.widget_decision_state:
    outcome_params: List[sight_pb2.DecisionParam] = []
    for key in sight.widget_decision_state['sum_outcome']:
      val = sight.widget_decision_state['sum_outcome'][key]
      if (utils.is_scalar(val)):
        #todo: assuming only double for now in scalar
        value = sight_pb2.Value(
            sub_type=sight_pb2.Value.ST_DOUBLE,
            double_value=val,
        )
      else:
        if (isinstance(val, dict) or isinstance(val, list)):
          json_value = json.dumps(val)
        elif (isinstance(val, pd.Series)):
          json_value = json.dumps(val.to_dict())
        else:
          raise TypeError(f'Value of {key} needs to be dict, list or pd.Series type. Actual type is {type(val)}, val={val}.')

        value = sight_pb2.Value(sub_type=sight_pb2.Value.ST_JSON,
                                json_value=json_value)

      outcome_params.append(sight_pb2.DecisionParam(
          key=key,
          value=value,
      ))
    decision_outcome.outcome_params.extend(outcome_params)

  if 'discount' in sight.widget_decision_state:
    decision_outcome.discount = sight.widget_decision_state['discount']

  return decision_outcome


def decision_point(
    choice_label: str,
    sight: Any,
) -> Dict[Text, float]:
  """Documents an execution point when a decision is made.

  If chosen_option is not provided, it is logged into sight. Otherwise, this
  method uses its own decision procedure, guided by the previously observed
  decisions and their outcomes, to make a choice and returns the corresponding
  chosen_option and parameters.

  Args:
    choice_label: Identifies the choice being made.
    sight: Instance of a Sight logger.

  Returns:
    Dict that maps the name of each action variable to its chosen value.
  """
  method_name = 'decision_point'
  logging.debug('>>>>>>>>>  In %s of %s', method_name, _file_name)
  # logging.info('>>>>>>>>>  In %s of %s, sight.widget_decision_state=%s', method_name, _file_name, sight.widget_decision_state)

  sight.widget_decision_state['num_decision_points'] += 1
  chosen_action = None

  if 'constant_action' in sight.widget_decision_state:
    return sight.widget_decision_state['constant_action']

  req = service_pb2.DecisionPointRequest()

  if FLAGS.deployment_mode == 'local' or _TRAINED_MODEL_LOG_ID.value:
    global _sight_id
    _sight_id = str(sight.id)
    client_id = str(sight.id)
    worker_location = '0'
  elif (FLAGS.deployment_mode == 'worker_mode'
        # or FLAGS.deployment_mode == 'docker_mode'
       ):
    client_id = os.environ['PARENT_LOG_ID']
    worker_location = os.environ['worker_location']

  req.client_id = client_id
  req.worker_id = f'client_{client_id}_worker_{worker_location}'

  if _OPTIMIZER_TYPE.value == 'dm_acme':
    optimizer_obj = optimizer.get_instance()
    selected_action = optimizer_obj.decision_point(sight, req)
    # print("selected_action : ", selected_action, type(selected_action), selected_action.shape, )
    # raise SystemError

    chosen_action = {}
    #? when action space is scalar (DQN agent - cartpole)
    if (selected_action.shape == ()):
      chosen_action[sight.widget_decision_state['decision_episode_fn'].
                    action_attrs[0]] = selected_action[()]
    #? when action space is 1d array (D4pg agent - pendulum)
    else:
      for i in range(
          len(sight.widget_decision_state['decision_episode_fn'].action_attrs)):
        chosen_action[sight.widget_decision_state['decision_episode_fn'].
                      action_attrs[i]] = selected_action[i]
    # print("chosen_action : ", chosen_action)

  # selected_action will be same for all calls of decision point in these
  # optimizers. As such, it is cached as the constant action.
  elif _OPTIMIZER_TYPE.value in [
      'vizier', 'genetic_algorithm', 'exhaustive_search', 'bayesian_opt',
      'sensitivity_analysis', 'smcpy'
  ] or _OPTIMIZER_TYPE.value.startswith('ng_'):
    optimizer_obj = optimizer.get_instance()
    chosen_action = optimizer_obj.decision_point(sight, req)
    sight.widget_decision_state['constant_action'] = chosen_action
  elif _OPTIMIZER_TYPE.value == 'worklist_scheduler':
    if (not optimizer.obj):
      optimizer.obj = SingleActionOptimizerClient(
          sight_pb2.DecisionConfigurationStart.OptimizerType.
          OT_WORKLIST_SCHEDULER, sight)
    optimizer_obj = optimizer.get_instance()
    chosen_action = optimizer_obj.decision_point(sight, req)
    # if(chosen_action == None):
    #     print("received None in chosen action")
    #     return None
    sight.widget_decision_state['constant_action'] = chosen_action

  elif _OPTIMIZER_TYPE.value.startswith('llm_'):
    optimizer_obj = optimizer.get_instance()
    if 'reward' in sight.widget_decision_state:
      req.decision_outcome.reward = sight.widget_decision_state['reward']
    if 'outcome_value' in sight.widget_decision_state:
      outcome_params: List[sight_pb2.DecisionParam] = []
      for key in sight.widget_decision_state['outcome_value']:
        outcome_params.append(
            sight_pb2.DecisionParam(
                key=key,
                value=sight_pb2.Value(
                    sub_type=sight_pb2.Value.ST_DOUBLE,
                    double_value=sight.widget_decision_state['outcome_value']
                    [key],
                ),
            ))
      req.decision_outcome.outcome_params.extend(outcome_params)
    req.decision_outcome.discount = sight.widget_decision_state['discount']
    chosen_action = optimizer_obj.decision_point(sight, req)

  choice_params: List[sight_pb2.DecisionParam] = []
  # for attr in sight.widget_decision_state[
  #         'decision_episode_fn'].action_attrs:
  for attr in chosen_action.keys():
    #? keep this might need to change sub_type of deicision param value
    if isinstance(chosen_action[attr], str):
      val = sight_pb2.Value(
          sub_type=sight_pb2.Value.ST_STRING,
          string_value=chosen_action[attr],
      )
    elif isinstance(chosen_action[attr], float):
      val = sight_pb2.Value(
          sub_type=sight_pb2.Value.ST_DOUBLE,
          double_value=chosen_action[attr],
      )
    elif isinstance(chosen_action[attr], int):
        val = sight_pb2.Value(
            sub_type=sight_pb2.Value.ST_INT64,
            int64_value=chosen_action[attr],
        )
    else:
      raise ValueError("unsupported type!!")

    choice_params.append(sight_pb2.DecisionParam(
        key=attr,
        value=val,
    ))

  # pytype: disable=attribute-error
  obj = sight_pb2.Object(
      sub_type=sight_pb2.Object.ST_DECISION_POINT,
      decision_point=sight_pb2.DecisionPoint(choice_label=choice_label,
                                             # choice_params=choice_params,
                                            ),
  )
  obj.decision_point.choice_params.extend(choice_params)
  sight.log_object(obj, inspect.currentframe().f_back.f_back)

  logging.info('decision_point() chosen_action=%s', chosen_action)
  logging.debug('<<<< Out %s of %s', method_name, _file_name)
  return chosen_action


def decision_outcome(
    outcome_label: str,
    sight: Any,
    reward: Optional[float] = None,
    outcome: Optional[Dict[str, Any]] = None,
    discount=1.0,
    # optimizer_type: str
) -> None:
  """Documents the outcome of prior decisions.

  Args:
    outcome_label: Label that identifies the outcome.
    sight: Instance of a Sight logger.
    reward: The numeric value of the quality of this outcome, with higher values being
      more desirable.
    outcome: Dictionary that describes the various outcome attributes of the application.
    discount: discount value to be used
  """
  method_name = 'decision_outcome'
  logging.debug('>>>>>>>>>  In %s of %s', method_name, _file_name)

  sight.widget_decision_state['discount'] = discount

  if reward is not None:
    logging.info('decision_outcome() reward=%s', reward)
    sight.widget_decision_state['reward'] = reward
    if 'sum_reward' not in sight.widget_decision_state:
      sight.widget_decision_state['sum_reward'] = 0
    sight.widget_decision_state['sum_reward'] += reward

  if outcome is not None:
    logging.info('decision_outcome() outcome=%s', outcome)
    if 'sum_outcome' not in sight.widget_decision_state:
      sight.widget_decision_state['sum_outcome'] = {}
    for key in outcome:
      # print(key, outcome[key], type(outcome[key]))
      # checking for scalar types
      if utils.is_scalar(outcome[key]):
        if key not in sight.widget_decision_state['sum_outcome']:
          sight.widget_decision_state['sum_outcome'][key] = 0
        sight.widget_decision_state['sum_outcome'][key] += outcome[key]
      # converting json into string
      else:
        # converting pandas datafram to json and storing it as json string
        # sight.widget_decision_state['sum_outcome'][key] = json.dumps(outcome[key].to_json())
        sight.widget_decision_state['sum_outcome'][key] = outcome[key]

      # if not isinstance(outcome[key], float) and not isinstance(outcome[key], int):
      #   continue
      # if key not in sight.widget_decision_state['sum_outcome']:
      #   sight.widget_decision_state['sum_outcome'][key] = 0
      # sight.widget_decision_state['sum_outcome'][key] += outcome[key]

  sight.log_object(
      sight_pb2.Object(
          sub_type=sight_pb2.Object.ST_DECISION_OUTCOME,
          decision_outcome=get_decision_outcome_proto(outcome_label, sight),
      ),
      inspect.currentframe().f_back.f_back,
  )

  logging.debug("<<<<  Out %s of %s", method_name, _file_name)


def propose_actions(sight, action_dict):
  request = service_pb2.ProposeActionRequest()
  if sight.params.silent_logger:
    raise ValueError('Cannot use Decision API using Sight silent logger.')
  request.client_id = str(sight.id)

  actions_data = []
  attributes_data = []

  # Process actions
  for k, v in action_dict.items():
    action_attr = sight_pb2.DecisionParam()
    action_attr.key = k
    if isinstance(v, str):
      val = sight_pb2.Value(
          sub_type=sight_pb2.Value.ST_STRING,
          string_value=v,
      )
    else:
      val = sight_pb2.Value(
          sub_type=sight_pb2.Value.ST_DOUBLE,
          double_value=v,
      )
    action_attr.value.CopyFrom(val)
    # Append to actions_data list
    actions_data.append(action_attr)
  request.action_attrs.extend(actions_data)

  attr_dict = sight.fetch_attributes()
  # print('attr_dict : ', attr_dict)

  # Process attributes
  for k, v in attr_dict.items():
    attribute = sight_pb2.DecisionParam()
    attribute.key = k
    if isinstance(v, str):
      val = sight_pb2.Value(
          sub_type=sight_pb2.Value.ST_STRING,
          string_value=v,
      )
    else:
      val = sight_pb2.Value(
          sub_type=sight_pb2.Value.ST_DOUBLE,
          double_value=v,
      )
    attribute.value.CopyFrom(val)
    # Append to attributes_data list
    attributes_data.append(attribute)
    request.attributes.extend(attributes_data)

  response = service.call(
      lambda s, meta: s.ProposeAction(request, 300, metadata=meta))
  action_id = response.action_id

  # log_object call
  sight_obj = sight_pb2.Object()
  sight_obj.sub_type = sight_pb2.Object.SubType.ST_PROPOSE_ACTION
  sight_obj.propose_action.action_id = str(action_id)
  sight_obj.propose_action.action_attrs.extend(actions_data)
  sight_obj.propose_action.attributes.extend(attributes_data)

  frame = inspect.currentframe().f_back.f_back
  sight.set_object_code_loc(sight_obj, frame)
  sight.log_object(sight_obj, True)

  return action_id


def finalize_episode(sight):  # , optimizer_obj
  """Finalize the run.

  Args:
    sight: Instance of a Sight logger.
    optimizer_obj: Object of Optimizer instance
  """
  method_name = 'finalize_episode'
  logging.debug('>>>>>>>>>  In %s of %s', method_name, _file_name)

  if (FLAGS.deployment_mode == 'local'
      # or FLAGS.deployment_mode == 'docker_mode'
      or FLAGS.deployment_mode == 'worker_mode'):
    if FLAGS.deployment_mode == 'local':
      client_id = str(sight.id)
      worker_location = '0'
    elif (FLAGS.deployment_mode == 'worker_mode'
          # or FLAGS.deployment_mode == 'docker_mode'
         ):
      client_id = os.environ['PARENT_LOG_ID']
      worker_location = os.environ['worker_location']

    req = service_pb2.FinalizeEpisodeRequest(
        client_id=client_id,
        worker_id=f'client_{client_id}_worker_{worker_location}',
    )

    if _OPTIMIZER_TYPE.value in [
        'genetic_algorithm', 'exhaustive_search', 'vizier', 'bayesian_opt',
        'sensitivity_analysis', 'smcpy'
    ] or _OPTIMIZER_TYPE.value.startswith(
        'llm_') or _OPTIMIZER_TYPE.value.startswith('ng_'):
      req.decision_outcome.CopyFrom(get_decision_outcome_proto(
          'outcome', sight))
      optimizer_obj = optimizer.get_instance()
      optimizer_obj.finalize_episode(sight, req)
    elif _OPTIMIZER_TYPE.value == 'worklist_scheduler':
      if (not optimizer.obj):
        optimizer.obj = SingleActionOptimizerClient(
            sight_pb2.DecisionConfigurationStart.OptimizerType.
            OT_WORKLIST_SCHEDULER, sight)
      req.decision_outcome.CopyFrom(
          get_decision_outcome_proto('outcome', sight))
      optimizer_obj = optimizer.get_instance()
      optimizer_obj.finalize_episode(sight, req)
    elif _OPTIMIZER_TYPE.value == 'dm_acme':
      optimizer_obj = optimizer.get_instance()
      optimizer_obj.finalize_episode(sight)

    if 'outcome_value' in sight.widget_decision_state:
      del sight.widget_decision_state['outcome_value']

  else:
    logging.info('Not in local/worker mode, so skipping it')

    if sight.widget_decision_state['proposed_actions']:
      for proposal in sight.widget_decision_state['proposed_actions']:
        # logging.info('proposal=%s', proposal)
        proposal_req = service_pb2.ProposeActionRequest(
            client_id=client_id,
            worker_id=f'client_{client_id}_worker_{worker_location}',
            outcome=sight_pb2.DecisionOutcome(
                outcome_label='estimated_outcome',
                outcome_value=proposal['outcome'],
            ),
            action=proposal['action'],
        )

        response = service.call(
            lambda s, meta: s.ProposeAction(proposal_req, 300, metadata=meta))
      sight.widget_decision_state['proposed_actions'] = []

  if 'sum_reward' in sight.widget_decision_state:
    _rewards.append(sight.widget_decision_state['sum_reward'])
  sight.widget_decision_state.pop('sum_reward', None)
  sight.widget_decision_state.pop('sum_outcome', None)

  logging.debug("<<<<  Out %s of %s", method_name, _file_name)


def get_outcome(sight):
  request = service_pb2.GetOutcomeRequest()
  request.client_id = str(sight.id)
  # request.unique_ids.append(3)
  response = service.call(
      lambda s, meta: s.GetOutcome(request, 300, metadata=meta))

  if (response.response_str):
    return response.response_str

  outcome_list = []
  for outcome in response.outcome:
    outcome_dict = {}
    outcome_dict['reward'] = outcome.reward
    outcome_dict['action'] = dict(outcome.action_attrs)
    outcome_dict['outcome'] = dict(outcome.outcome_attrs)
    outcome_list.append(outcome_dict)
  return outcome_list


def finalize(sight):
  logging.info(
      'Get latest status of this training by running this script : '
      'python3 x-sight/py/sight/widgets/decision/current_status.py'
      ' --log_id=%s --service_name=%s',
      sight.id,
      service.get_service_id(),
  )
