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
import os
import sys
import dm_env
import numpy as np
from typing import Any, Callable, Dict, List, Optional, Text

from absl import flags
# import logging
import logging


from sight.service_utils import service_pb2
from sight import service_utils as service
from sight.proto import sight_pb2
from sight.widgets.decision.llm_optimizer_client import LLMOptimizerClient
from sight.widgets.decision.single_action_optimizer_client import SingleActionOptimizerClient
from sight.widgets.decision.acme.acme_optimizer_client import AcmeOptimizerClient
from sight.widgets.decision.env_driver import driver_fn
# from sight.widgets.decision.cartpole_driver import driver_fn
from sight.widgets.decision import decision_episode_fn
from sight.widgets.decision import trials

logging.basicConfig(level=logging.DEBUG)

_DECISON_MODE = flags.DEFINE_enum(
    'decision_mode',
    None,
    ['train', 'run', 'configured_run'],
    (
        'Indicates whether the decision API should be used to train a decision '
        'model (train) or use it to run the application (run).'
    ),
)
_DECISION_RUN_CONFIG_FILE = flags.DEFINE_string(
    'decision_run_config_file',
    None,
    (
        'File that contains the Sight configuration for driving this '
        "application's execution via the Decision API"
    ),
)
_DECISION_PARAMS = flags.DEFINE_string(
    'decision_params',
    None,
    (
        'Assignment of the action attributes for the decision. Format: '
        + '"key1=val1:key2=val2:...".'
    ),
)
_DEPLOYMENT_MODE = flags.DEFINE_enum(
    'deployment_mode',
    None,
    ['distributed', 'dsub_local', 'docker_local', 'local', 'worker_mode'],
    (
        'The procedure to use when training a model to drive applications that '
        'use the Decision API.'
    ),
)
_OPTIMIZER_TYPE = flags.DEFINE_enum(
    'optimizer_type',
    None,
    ['vizier', 'dm_acme', 'genetic_algorithm', 'exhaustive_search',
     'llm_text_bison', 'llm_chat_bison', 'llm_gemini_pro'],
    'The optimizer to use',
)
_NUM_TRAIN_WORKERS = flags.DEFINE_integer(
    'num_train_workers', 1, 'Number of workers to use in a training run.'
)
_NUM_TRIALS = flags.DEFINE_integer(
    'num_trials', None, 'Number of trials to perform.'
)
_DECISION_TRAIN_OUT_CONFIG_FILE = flags.DEFINE_string(
    'decision_train_out_config_file',
    None,
    (
        'File to which the configuration that describes the trained model will '
        'be written.'
    ),
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
    ['CartPole-v1', 'MountainCar-v0', 'Acrobot-v1', 'Pendulum-v1', 'None'],
    'What environment to run',
)

_TRAINED_MODEL_LOG_ID = flags.DEFINE_string(
  'trained_model_log_id',
  None,
  'Sight log Id of trained run to be used'
)

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
        decision_configuration.choice_config
    )

  if 'state' not in widget_decision_state:
    widget_decision_state['state'] = {}

  if 'decision_episode_fn' not in widget_decision_state:
    widget_decision_state['decision_episode_fn'] = None

  if 'rl_decision_driver' not in widget_decision_state:
    widget_decision_state['rl_decision_driver'] = None

  logging.debug("<<<<  Out %s of %s", method_name, _file_name)


def _attr_dict_to_proto(
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
            valid_int_values=valid_values
        )

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
      minimum = np.repeat(array.minimum, array.shape[0]) if array.minimum.size == 1 else array.minimum
      maximum = np.repeat(array.maximum, array.shape[0]) if array.maximum.size == 1 else array.maximum

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
  # print(result)

  return result


def run(
    sight: Any,
    env: Any = None,
    driver_fn: Callable[[Any], Any] = driver_fn,
    state_attrs: Dict[str, sight_pb2.DecisionConfigurationStart.AttrProps] = {},
    action_attrs: Dict[
        str, sight_pb2.DecisionConfigurationStart.AttrProps
    ] = {},
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
    description: Human-readable description of the application.
  """

  method_name = 'run'
  logging.debug('>>>>>>>>>  In %s of %s', method_name, _file_name)

  # when gym env passed, fetch state and action from env directly
  if state_attrs == {}:
    state_attrs = attr_to_dict('state', env.observation_spec())
  # print('state_attrs : ', state_attrs)
  if action_attrs == {}:
    action_attrs = attr_to_dict('action', env.action_spec())
  # print('action_Attr : ', action_attrs)

  sight.widget_decision_state['decision_episode_fn'] = (
      decision_episode_fn.DecisionEpisodeFn(
          driver_fn, state_attrs, action_attrs
      )
  )
  # print(sight.widget_decision_state['decision_episode_fn'])
  # raise SystemError

  if _OPTIMIZER_TYPE.value == 'dm_acme':
    optimizer.obj = AcmeOptimizerClient(sight)
  elif _OPTIMIZER_TYPE.value == 'vizier':
    optimizer.obj = SingleActionOptimizerClient(sight_pb2.DecisionConfigurationStart.OptimizerType.OT_VIZIER, sight)
  elif _OPTIMIZER_TYPE.value == 'genetic_algorithm':
    optimizer.obj = GeneticAlgorithmOptimizerClient(
      max_population_size = _NUM_TRAIN_WORKERS.value, sight=sight)
  elif _OPTIMIZER_TYPE.value == 'exhaustive_search':
    optimizer.obj = SingleActionOptimizerClient(sight_pb2.DecisionConfigurationStart.OptimizerType.OT_EXHAUSTIVE_SEARCH, sight)
  elif _OPTIMIZER_TYPE.value.startswith('llm_'):
    optimizer.obj = LLMOptimizerClient(_OPTIMIZER_TYPE.value.partition('llm_')[2], description, sight)
  else:
    raise ValueError(f'Unknown optimizer type {_OPTIMIZER_TYPE.value}')



  decision_configuration = sight_pb2.DecisionConfigurationStart()
  decision_configuration.optimizer_type = optimizer.obj.optimizer_type()

  decision_configuration.choice_config[sight.params.label].CopyFrom(optimizer.obj.create_config())
  _attr_dict_to_proto(state_attrs, decision_configuration.state_attrs)
  _attr_dict_to_proto(action_attrs, decision_configuration.action_attrs)
  # print('decision_config : ', decision_configuration)
  # raise SystemError


  sight.enter_block(
      'Decision Configuration',
      sight_pb2.Object(
          block_start=sight_pb2.BlockStart(
              sub_type=sight_pb2.BlockStart.ST_CONFIGURATION,
              configuration=sight_pb2.ConfigurationStart(
                  sub_type=sight_pb2.ConfigurationStart.ST_DECISION_CONFIGURATION,
                  decision_configuration=decision_configuration,
              ),
          )
      ),
  )
  sight.exit_block('Decision Configuration', sight_pb2.Object())
  sight.widget_decision_state['num_decision_points'] = 0

  # sight.widget_decision_state['decision_episode_fn'] = (
  #     decision_episode_fn.DecisionEpisodeFn(
  #         driver_fn, state_attrs, action_attrs
  #     )
  # )

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

    if(not FLAGS.trained_model_log_id):
      raise ValueError("trained_model_log_id have to be passed from the trained run for decision_mokde = run")

    req = service_pb2.FetchOptimalActionRequest(
        client_id=FLAGS.trained_model_log_id,
        # worker_id=f'client_{client_id}_worker_{worker_location}',
    )
    response = service.call(
        lambda s, meta: s.FetchOptimalAction(req, 300, metadata=meta)
    )
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
      sight.widget_decision_state['sum_outcome'] = 0
      sight.widget_decision_state['last_reward'] = None
    else:
      raise ValueError(
          'In configured_run mode decision_run_config_file is required.'
      )

    # If a docker image is provided, run within it.
    logging.info(
        'decision_train_alg=%s docker_image=%s',
        _DEPLOYMENT_MODE.value,
        _DOCKER_IMAGE.value,
    )
    if _DEPLOYMENT_MODE.value == 'local' and _DOCKER_IMAGE.value:
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
    # print('details.action_min : ', details.action_min.values(), list(details.action_min.values())[0])

    if(_OPTIMIZER_TYPE.value == 'exhaustive_search'):
      possible_actions = list(details.action_max.values())[0] - list(details.action_min.values())[0] + 2
      print(possible_actions)
      if(_OPTIMIZER_TYPE.value == 'exhaustive_search' and possible_actions < (_NUM_TRIALS.value or int(os.environ['num_samples']))):
        raise ValueError(f"max possible value for num_trials is : {possible_actions}")

    print('_DECISON_MODE.value : ', _DECISON_MODE.value)
    if _DEPLOYMENT_MODE.value == 'distributed':
      logging.info('_DEPLOYMENT_MODE.value == distributed')
      if(not _DOCKER_IMAGE.value):
        raise ValueError("docker_image must be provided for distributed mode")
      print("decision_config : ", decision_configuration)
      trials.launch(
          optimizer.obj,
          decision_configuration,
          _NUM_TRAIN_WORKERS.value,
          sight,
      )
      trials.start_jobs(
          _NUM_TRAIN_WORKERS.value,
          _NUM_TRIALS.value,
          _BINARY_PATH.value,
          _OPTIMIZER_TYPE.value,
          _DOCKER_IMAGE.value,
          _DECISON_MODE.value,
          'worker_mode',
          'dsub_cloud_worker',
          sight,
      )
    elif _DEPLOYMENT_MODE.value in [
        'local',
        'dsub_local',
        'docker_local',
        'worker_mode',
    ]:
      if _DEPLOYMENT_MODE.value == 'worker_mode':
        num_samples_to_run = int(os.environ['num_samples'])
      else:
        trials.launch(
            optimizer.obj,
            decision_configuration,
            _NUM_TRAIN_WORKERS.value,
            sight,
        )
        num_samples_to_run = _NUM_TRIALS.value

      # If a docker image is provided, run within it.
      if (
          _DEPLOYMENT_MODE.value == 'docker_local'
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
      elif (
          _DEPLOYMENT_MODE.value == 'dsub_local'
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

        for _ in range(num_samples_to_run):
          sight.enter_block('Decision Sample', sight_pb2.Object())
          if 'constant_action' in sight.widget_decision_state:
            del sight.widget_decision_state['constant_action']
          sight.widget_decision_state['sum_outcome'] = 0
          sight.widget_decision_state['last_reward'] = None

          if env:
            driver_fn(env, sight)
            # user have to pass dm_env type environment
          else:
            driver_fn(sight)

          finalize_episode(sight)
          sight.exit_block('Decision Sample', sight_pb2.Object())

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
  if (
      sight.widget_decision_state is not None
      and 'decision_episode_fn' in sight.widget_decision_state
      and sight.widget_decision_state['decision_episode_fn']
      and name in sight.widget_decision_state['decision_episode_fn'].state_attrs
  ):
    sight.widget_decision_state['state'][name] = obj_to_log


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

  sight.widget_decision_state['num_decision_points'] += 1
  chosen_action = None

  if 'constant_action' in sight.widget_decision_state:
    return sight.widget_decision_state['constant_action']

  req = service_pb2.DecisionPointRequest()

  if _DEPLOYMENT_MODE.value == 'local' or _TRAINED_MODEL_LOG_ID.value:
    global _sight_id
    _sight_id = str(sight.id)
    client_id = str(sight.id)
    worker_location = '0'
  elif (
      _DEPLOYMENT_MODE.value
      == 'worker_mode'
      # or _DEPLOYMENT_MODE.value == 'docker_mode'
  ):
    client_id = os.environ['PARENT_LOG_ID']
    worker_location = os.environ['worker_location']

  # req.client_id = client_id
  # req.worker_id = f'client_{client_id}_worker_{worker_location}'

  if _OPTIMIZER_TYPE.value == 'dm_acme':

    optimizer_obj = optimizer.get_instance()
    selected_action = optimizer_obj.decision_point(sight, req)
    # print("selected_action : ", selected_action, type(selected_action), selected_action.shape, )
    # raise SystemError

    chosen_action = {}
    #? when action space is scalar (DQN agent - cartpole)
    if(selected_action.shape == ()):
      chosen_action[
            sight.widget_decision_state['decision_episode_fn'].action_attrs[0]
        ] = selected_action[()]
    #? when action space is 1d array (D4pg agent - pendulum)
    else:
      for i in range(len(sight.widget_decision_state['decision_episode_fn'].action_attrs)):
        chosen_action[
            sight.widget_decision_state['decision_episode_fn'].action_attrs[i]
        ] = selected_action[i]
    # print("chosen_action : ", chosen_action)

  # selected_action will be same for all calls of decision point in these
  # optimizers. As such, it is cached as the constant action.
  elif _OPTIMIZER_TYPE.value in ['vizier', 'genetic_algorithm', 'exhaustive_search']:

    optimizer_obj = optimizer.get_instance()
    chosen_action = optimizer_obj.decision_point(sight, req)

    sight.widget_decision_state['constant_action'] = chosen_action
  elif _OPTIMIZER_TYPE.value.startswith('llm_'):
    optimizer_obj = optimizer.get_instance()
    # logging.info('sight.widget_decision_state=%s', sight.widget_decision_state)
    if 'outcome_value' in sight.widget_decision_state:
      req.decision_outcome.outcome_value = sight.widget_decision_state['outcome_value']
      req.decision_outcome.discount = sight.widget_decision_state['discount']
    chosen_action = optimizer_obj.decision_point(sight, req)

  #? keep this might need to change sub_type of deicision param value
  choice_params: List[sight_pb2.DecisionParam] = []
  for attr in sight.widget_decision_state['decision_episode_fn'].action_attrs:
    choice_params.append(
        sight_pb2.DecisionParam(
            key=attr,
            value=sight_pb2.Value(
                sub_type=sight_pb2.Value.ST_DOUBLE,
                double_value=chosen_action[attr],
            ),
        )
    )
  # pytype: disable=attribute-error
  obj = sight_pb2.Object(
      sub_type=sight_pb2.Object.ST_DECISION_POINT,
      decision_point=sight_pb2.DecisionPoint(
          choice_label=choice_label,
          # choice_params=choice_params,
      ),
  )
  obj.decision_point.choice_params.extend(choice_params)
  sight.log_object(obj, inspect.currentframe().f_back.f_back)

  logging.debug('<<<< Out %s of %s', method_name, _file_name)
  return selected_action



def decision_outcome(
    outcome_label: str,
    outcome_value: float,
    sight: Any,
    discount=1.0,
    # optimizer_type: str
) -> None:
  """Documents the outcome of prior decisions.

  Args:
    outcome_label: Label that identifies the outcome.
    outcome_value: The numeric value of this outcome, with higher values being
      more desirable.
    sight: Instance of a Sight logger.
    discount: discount value to be used
  """
  method_name = 'decision_outcome'
  logging.debug('>>>>>>>>>  In %s of %s', method_name, _file_name)

  sight.widget_decision_state['discount'] = discount
  sight.widget_decision_state['outcome_value'] = outcome_value
  sight.widget_decision_state['sum_outcome'] += outcome_value

  sight.log_object(
      sight_pb2.Object(
          sub_type=sight_pb2.Object.ST_DECISION_OUTCOME,
          decision_outcome=sight_pb2.DecisionOutcome(
              outcome_label=outcome_label, outcome_value=outcome_value
          ),
      ),
      inspect.currentframe().f_back.f_back,
  )

  logging.debug("<<<<  Out %s of %s", method_name, _file_name)


def propose_action(outcome: float, action: Dict[Text, float], sight) -> None:
  sight.widget_decision_state['proposed_actions'].append({
      'outcome': outcome,
      'action': dict(action),
  })


def finalize_episode(sight):  # , optimizer_obj
  """Finalize the run.

  Args:
    sight: Instance of a Sight logger.
    optimizer_obj: Object of Optimizer instance
  """
  method_name = 'finalize_episode'
  logging.debug('>>>>>>>>>  In %s of %s', method_name, _file_name)

  _rewards.append(round(sight.widget_decision_state['sum_outcome'], 2))
  print('rewards : ', _rewards, len(_rewards))

  if (
      _DEPLOYMENT_MODE.value == 'local'
      # or _DEPLOYMENT_MODE.value == 'docker_mode'
      or _DEPLOYMENT_MODE.value == 'worker_mode'
  ):
    if _DEPLOYMENT_MODE.value == 'local':
      client_id = str(sight.id)
      worker_location = '0'
    elif (
        _DEPLOYMENT_MODE.value
        == 'worker_mode'
        # or _DEPLOYMENT_MODE.value == 'docker_mode'
    ):
      client_id = os.environ['PARENT_LOG_ID']
      worker_location = os.environ['worker_location']

    req = service_pb2.FinalizeEpisodeRequest(
        client_id=client_id,
        worker_id=f'client_{client_id}_worker_{worker_location}',
    )

    if _OPTIMIZER_TYPE.value == 'vizier':
      decision_outcome = sight_pb2.DecisionOutcome(
          outcome_label='outcome',
          outcome_value=sight.widget_decision_state['sum_outcome'],
      )
      req.decision_outcome.CopyFrom(decision_outcome)
      optimizer_obj = optimizer.get_instance()
      optimizer_obj.finalize_episode(sight, req)
    elif _OPTIMIZER_TYPE.value in ['genetic_algorithm', 'exhaustive_search'] or _OPTIMIZER_TYPE.value.startswith('llm_'):
      decision_outcome = sight_pb2.DecisionOutcome(
          outcome_label='outcome',
          outcome_value=sight.widget_decision_state['sum_outcome'],
      )
      req.decision_outcome.CopyFrom(decision_outcome)
    elif _OPTIMIZER_TYPE.value == 'dm_acme':
      optimizer_obj = optimizer.get_instance()
      optimizer_obj.finalize_episode(sight, req)

    #! not calling finalize episode to server
    # response = service.call(
    #     lambda s, meta: s.FinalizeEpisode(req, 300, metadata=meta)
    # )
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
          lambda s, meta: s.ProposeAction(proposal_req, 300, metadata=meta)
      )
    sight.widget_decision_state['proposed_actions'] = []

  logging.debug("<<<<  Out %s of %s", method_name, _file_name)


def finalize(sight):
  logging.info(
      'Get latest status of this training by running this script : '
      'python3 sight/widgets/decision/current_status.py'
      ' --log_id=%s --service_name=%s',
      sight.id,
      service.get_service_id(),
  )
