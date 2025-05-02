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
# import dm_env
import json
import os
from pathlib import Path
import random
import sys
import threading
import time
from typing import Any, Callable, Dict, List, Optional, Text, Union

from absl import flags
from dataclasses import dataclass
from dataclasses import dataclass
from google.protobuf.text_format import Merge
# from absl import logging
from helpers.logs.logs_handler import logger as logging
import numpy as np
import pandas as pd
from sight import service_utils as service
# from sight.sight import Sight
from sight.proto import sight_pb2
from sight.utility import poll_network_batch_outcome
from sight.utils.proto_conversion import convert_dict_to_proto
from sight.utils.proto_conversion import convert_proto_to_dict
# from sight.widgets.decision.cartpole_driver import driver_fn
from sight.widgets.decision import decision_episode_fn
from sight.widgets.decision import decision_helper
from sight.widgets.decision import trials
from sight.widgets.decision import utils
from sight.widgets.decision.env_driver import driver_fn
from sight.widgets.decision.utils import get_config_dir_path

# from sight.widgets.decision.acme.acme_optimizer_client import (
#     AcmeOptimizerClient
# )
# from sight.widgets.decision.env_driver import driver_fn
from sight.widgets.decision.llm_optimizer_client import LLMOptimizerClient
from sight.widgets.decision.single_action_optimizer_client import (
    SingleActionOptimizerClient)
from sight_service.proto import service_pb2
from sight_service.shared_batch_messages import CachedBatchMessages
from sight_service.shared_batch_messages import DecisionMessage

# logging.basicConfig(level=logging.DEBUG)

_DECISON_MODE = flags.DEFINE_enum(
    'decision_mode',
    'train',
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
        'vizier',
        'dm_acme',
        'genetic_algorithm',
        'exhaustive_search',
        'llm_text_bison_optimize',
        'llm_chat_bison_optimize',
        'llm_gemini_pro_optimize',
        'llm_text_bison_recommend',
        'llm_chat_bison_recommend',
        'llm_gemini_pro_recommend',
        'llm_text_bison_interactive',
        'llm_chat_bison_interactive',
        'llm_gemini_pro_interactive',
        'bayesian_opt',
        'sensitivity_analysis',
        'ng_auto',
        'ng_bo',
        'ng_cma',
        'ng_two_points_de',
        'ng_random_search',
        'ng_pso',
        'ng_scr_hammersley_search',
        'ng_de',
        'ng_cga',
        'ng_es',
        'ng_dl_opo',
        'ng_dde',
        'ng_nmm',
        'ng_tiny_spsa',
        'ng_voronoi_de',
        'ng_cma_small',
        'smcpy',
        'worklist_scheduler',
    ],
    'The optimizer to use',
)
_NUM_TRAIN_WORKERS = flags.DEFINE_integer(
    'num_train_workers', 1, 'Number of workers to use in a training run.')
_NUM_TRIALS = flags.DEFINE_integer('num_trials', 1,
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

_SERVER_QUEUE_BATCH_SIZE = flags.DEFINE_integer(
    'server_queue_batch_size',
    1,
    'batch size of the server queue for message queue',
)

_CACHE_MODE = flags.DEFINE_enum(
    'cache_mode', 'none',
    ['gcs', 'local', 'redis', 'none', 'gcs_with_redis', 'local_with_redis'],
    'Which Sight cache to use ? (default is none)')

_CONFIG_PATH = flags.DEFINE_string(
    'config_path', get_config_dir_path(), 'Directory location where the *_config.yaml files are stored')

_file_name = os.path.basename(__file__)
_rewards = []
FLAGS = flags.FLAGS

@dataclass
class DecisionConfig:
  """Configuration for the decision module."""
  questions: dict[str, Any]

  optimizers: dict[str, Any]

  workers: dict[str, Any]

  def __init__(self, config_dir_path: Optional[str] = None):
    """
    Arguments:
      config_dir_path: Path of the directory that contains the optimizer_config.yaml,
        worker_config.yaml and question_config.yaml files.
    """

    if(config_dir_path == None):
      raise ValueError("config_dir_path need to be passed and should contain all *_config.yaml files")

    self.questions = utils.load_yaml_config(
        config_dir_path + '/question_config.yaml'
    )
    self.optimizers = utils.load_yaml_config(
        config_dir_path + '/optimizer_config.yaml'
    )
    self.workers = utils.load_yaml_config(
        config_dir_path + '/worker_config.yaml'
    )
    # Load the configuration parameter's for this worker from the worker-specific
    # file path.
    for name, cfg in self.workers.items():
      self.workers[name] |= utils.load_yaml_config(cfg['file_path'])[cfg['version']]

def initialize(config: DecisionConfig, sight) -> None:
  """Initializes the decision module for this sight logger object.

  Arguments:
    config: The configuration of the decision module.
    sight: The sight object for which the decision module is being initialized.
  """

  for question_label, question_config in config.questions.items():
    if question_label not in config.optimizers:
      continue
    optimizer_type = config.optimizers[question_label]['optimizer']
    logging.info('optimizer_type=%s', optimizer_type)
    optimizer_config = config.optimizers[question_label]
    logging.info('optimizer_config=%s', optimizer_config)

    opt_obj = setup_optimizer(sight, optimizer_type)
    trials.launch(
        configure_decision(
          sight,
          question_label,
          question_config,
          optimizer_config,
          opt_obj
          ),
        sight)

    # Start worker jobs
    trials.start_worker_jobs(sight, question_label, optimizer_config, config.workers, optimizer_type)

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

  logging.debug('<<<<  Out %s of %s', method_name, _file_name)


def init_sight_polling_thread(sight_id, question_label):
  # print
  status_update_thread = threading.Thread(target=poll_network_batch_outcome,
                                          args=(sight_id, question_label))
  logging.info('*************** starting thread ************')
  status_update_thread.start()


class Optimizer:

  def __init__(self):
    self.obj = None

  def get_instance(self):
    return self.obj


optimizer = Optimizer()


def get_decision_messages_from_proto(
    decision_messages_proto: List[sight_pb2.DecisionMessage],
) -> Dict[str, Any]:
  messages = {}
  for msg in decision_messages_proto:
    messages[msg.action_id] = convert_proto_to_dict(proto=msg.action)
  return messages


def run(
    sight: Any,
    question_label: str = None,
    configs: Optional[DecisionConfig] = None,
    driver_fn: Callable[[Any], Any] = None,
    description: str = '',
    env: Any = None,
):
  """Driver for running applications that use the Decision API.
  """

  method_name = 'run'
  logging.debug('>>>>>>>>>  In %s of %s', method_name, _file_name)

  sight.widget_decision_state['num_decision_points'] = 0

  # only true if run function called from main script
  #! can we eliminate this condition by calling initialize from sight??
  if configs:
    # *** This configs are for each question's meta data

    for question_label, question_config in configs.questions.items():
      optimizer_config = configs.optimizers.get(question_label, None)

      optimizer_type = optimizer_config.get('optimizer', None)

      optimizer.obj = setup_optimizer(sight, optimizer_type)
      decision_configuration = configure_decision(sight, question_label,
                                                  question_config,
                                                  optimizer_config,
                                                  optimizer.obj)

      ## ! Don't know why we did this ?
      # sight.widget_decision_state['proposed_actions'] = []

      decision_mode_actions = {
          'run':
              execute_run_mode,
          # 'configured_run': (lambda sight=sight, driver_fn=driver_fn:
          #                    execute_configured_run_mode(sight, driver_fn)),
          'train': (
              lambda sight=sight, decision_configuration=decision_configuration,
              driver_fn=driver_fn, optimizer_config=optimizer_config,
              workers_config=configs.workers, optimizer_type=optimizer_type,
              question_label=question_label: execute_train_mode(
                  sight, decision_configuration, driver_fn, optimizer_config,
                  workers_config, optimizer_type, question_label)),
      }

      action = decision_mode_actions.get(_DECISON_MODE.value)
      if action:
        action()
      else:
        raise ValueError(f'Unknown decision mode {_DECISON_MODE.value}')
  # run function called from worker - without any config files
  else:
    optimizer.obj = setup_optimizer(sight, _OPTIMIZER_TYPE.value)
    client_id, worker_location = _configure_client_and_worker(sight=sight)
    num_retries = 0
    backoff_interval = 0.5
    while True:
      # #? new rpc just to check move forward or not?

      req = service_pb2.WorkerAliveRequest(
          client_id=client_id,
          worker_id=f'client_{client_id}_worker_{worker_location}',
          question_label=question_label)
      response = service.call(
          lambda s, meta: s.WorkerAlive(req, 300, metadata=meta))
      logging.info('Response from WorkerAlive RPC: %s', response)
      if (response.status_type ==
          service_pb2.WorkerAliveResponse.StatusType.ST_DONE):
        break
      elif (response.status_type ==
            service_pb2.WorkerAliveResponse.StatusType.ST_RETRY):
        # logging.info('Retrying in 5 seconds......')
        # time.sleep(5)
        backoff_interval *= 2
        time.sleep(random.uniform(backoff_interval / 2, backoff_interval))
        logging.info('backed off for %s seconds... and trying for %s',
                     backoff_interval, num_retries)
        num_retries += 1
        if (num_retries >= 50):
          break
      elif (response.status_type ==
            service_pb2.WorkerAliveResponse.StatusType.ST_ACT):
        process_worker_action(response, sight, driver_fn, env, question_label,
                              optimizer.obj)
      else:
        raise ValueError('Invalid response from server')

    logging.info('Exiting the training loop.')

  logging.debug('<<<<<< Exiting run method')

  logging.debug("<<<<  Out %s of %s", method_name, _file_name)


def execute_run_mode():
  """Executes the run mode.

  Raises:
    ValueError:
      If sight_log_id is not provided.
  """

  logging.info('_DECISON_MODE.value == run')
  if not FLAGS.sight_log_id:
    raise ValueError('sight_log_id must be provided for decision_mode = run')

  req = service_pb2.FetchOptimalActionRequest(
      client_id=FLAGS.sight_log_id,
      # worker_id=f'client_{client_id}_worker_{worker_location}',
  )
  response = service.call(
      lambda s, meta: s.FetchOptimalAction(req, 300, metadata=meta))
  print('response:', response.response_str)


#? commenting out this flow as it calls code to add config file from sight,
#  which uses capacitor file logic
# def execute_configured_run_mode(sight, driver_fn):
#   """Executes the configured run mode.

#   Args:
#     sight: The Sight object to be used for logging.
#     driver_fn: Driver function for calling application logic that uses the Sight
#       Decision API to describe decisions and their outcomes. It is assumed that
#       driver_fn does not maintain state across invocations and can be called as
#       many time as needed, possibly concurrently (i.e. does not keep state
#       within global variables either internally or via its interactions with
#       external resources).
#   """
#   if FLAGS.decision_run_config_file:
#     sight.add_config_file(_DECISION_RUN_CONFIG_FILE.value)
#   elif _DECISION_PARAMS.value:
#     chosen_action = {
#         key: float(val) for key, val in (
#             key_val.split('=') for key_val in _DECISION_PARAMS.value.split(':'))
#     }
#     sight.widget_decision_state['constant_action'] = chosen_action
#     sight.widget_decision_state['last_reward'] = None
#   else:
#     raise ValueError(
#         'In configured_run mode, decision_run_config_file is required.')

#   logging.info(
#       'decision_train_alg=%s docker_image=%s',
#       FLAGS.server_mode,
#       _DOCKER_IMAGE.value,
#   )

#   if FLAGS.server_mode == 'local' and _DOCKER_IMAGE.value:
#     trials.start_job_in_docker(
#         1,
#         _BINARY_PATH.value,
#         _OPTIMIZER_TYPE.value,
#         _DOCKER_IMAGE.value,
#         _DECISON_MODE.value,
#         'docker_worker',
#         'worker_mode',
#         _DECISION_PARAMS.value,
#         sight,
#     )
#   else:
#     driver_fn(sight)


def execute_train_mode(sight, decision_configuration, driver_fn,
                       optimizer_config, workers_config, optimizer_type,
                       question_label):
  """Executes the train mode.
  """
  # validate_train_mode(sight)
  # if FLAGS.server_mode in ['cloud_run', 'vm']:
  if FLAGS.worker_mode is None:
    create_opt_and_start_workers(sight, question_label, decision_configuration,
                                 optimizer_config, workers_config,
                                 optimizer_type)
  # elif FLAGS.server_mode in [
  #     'local',
  #     'dsub_local',
  #     'docker_local',
  #     'worker_mode',
  # ]:
  # elif FLAGS.worker_mode in [
  #     'dsub_local_worker',
  #     'dsub_cloud_worker',
  # ]:
  #   execute_local_training(sight, decision_configuration, driver_fn,
  #                          optimizer_config, workers_config, optimizer_type)
  else:
    raise ValueError(f'worker mode is not None')


# def validate_train_mode(sight):
#   if FLAGS.server_mode in ['cloud_run', 'vm']:
#     #! check if we need this condition now or not
#     # details = sight.widget_decision_state['decision_episode_fn']
#     # print("details : ", details)
#     # possible_actions = (list(details.action_max.values())[0] -
#     #                     list(details.action_min.values())[0] + 2)
#     # if (_OPTIMIZER_TYPE.value == 'exhaustive_search' and
#     #     possible_actions < _NUM_TRIALS.value):
#     #   raise ValueError(
#     #       f'Max possible value for num_trials is: {possible_actions}')
#     if not _DOCKER_IMAGE.value:
#       raise ValueError('docker_image must be provided for distributed mode')


# def execute_local_training(sight, decision_configuration, driver_fn,
#                            optimizer_config, workers_config, optimizer_type):
#   """Executes the local training mode.
#   """
#   if FLAGS.worker_mode or 'PARENT_LOG_ID' in os.environ:
#     pass
#   else:
#     trials.launch(
#         decision_configuration,
#         sight,
#     )

#   # if FLAGS.server_mode == 'docker_local':
#   #   trials.start_job_in_docker(
#   #       _NUM_TRIALS.value,
#   #       _BINARY_PATH.value,
#   #       _OPTIMIZER_TYPE.value,
#   #       _DOCKER_IMAGE.value,
#   #       _DECISON_MODE.value,
#   #       'worker_mode',
#   #       'docker_local_worker',
#   #       _DECISION_PARAMS.value,
#   #       sight,
#   #   )
#   if FLAGS.server_mode == 'dsub_local':
#     trials.start_worker_jobs(sight, optimizer_config, workers_config,
#                              optimizer_type)


def process_worker_action(response, sight, driver_fn, env, question_label,
                          opt_obj):
  """Processes worker actions during local training.

  Args:
      response: The response from the WorkerAlive RPC.
      sight: Sight object used for logging and configuration.
      driver_fn: The driver function that drives the training.
      env: The environment in which the training takes place (optional).
      question_label:
  """
  decision_messages = get_decision_messages_from_proto(
      decision_messages_proto=response.decision_messages)
  # shared_batch_messages = CachedBatchMessages()
  sight.widget_decision_state['cached_messages'] = opt_obj.cache
  logging.info('cached_messages=%s',
               sight.widget_decision_state['cached_messages'])

  for action_id, action_params in decision_messages.items():
    logging.info('action_id=%s, action_params=%s', action_id, action_params)
    sight.enter_block('Decision Sample', sight_pb2.Object())

    if 'constant_action' in sight.widget_decision_state:
      del sight.widget_decision_state['constant_action']

    cached_messages = sight.widget_decision_state['cached_messages']
    sight.widget_decision_state['discount'] = 0
    sight.widget_decision_state['last_reward'] = None
    sight.widget_decision_state['action_id'] = action_id

    cached_messages.set(
        action_id,
        DecisionMessage(
            action_id=action_id,
            action_params=action_params,
        ),
    )

    if env:
      driver_fn(env, sight)
    else:
      driver_fn(sight)

    sight.exit_block('Decision Sample', sight_pb2.Object())

  finalize_episode(question_label, sight)


def create_opt_and_start_workers(sight, question_label, decision_configuration,
                                 optimizer_config, workers_config,
                                 optimizer_type):
  """Executes the distributed training mode.

  Args:
    sight: The Sight object to be used for logging.
    decision_configuration: The decision configuration proto.
  """
  trials.launch(decision_configuration, sight)
  trials.start_worker_jobs(sight, question_label, optimizer_config, workers_config,
                           optimizer_type)


def get_decision_configuration_for_opt(
    sight, question_label, opt_obj, question_config,
    optimizer_config) -> sight_pb2.DecisionConfigurationStart:
  """Preparing decision_configuration for optimizer to be used in server

  Args:
      sight : The Sight object to be used for logging.
      question_label : The label associated with the current question.
        opt_obj : The optimizer object that creates a choice configuration.
      question_config : The configuration dictionary for the current question.
      optimizer_config : The configuration dictionary for the optimizer.

  Returns:
      decision_configuration: The decision configuration protobuf object with optimizer configuration.
  """
  relative_text_proto_path = question_config['attrs_text_proto']
  if os.path.exists(relative_text_proto_path):
    with open(relative_text_proto_path, 'r') as f:
      text_proto_data = f.read()
  else:
    current_file = Path(__file__).resolve()
    sight_repo_path = current_file.parents[4]
  relative_text_proto_path = question_config['attrs_text_proto']
  if os.path.exists(relative_text_proto_path):
    with open(relative_text_proto_path, 'r') as f:
      text_proto_data = f.read()
  else:
    current_file = Path(__file__).resolve()
    sight_repo_path = current_file.parents[4]

    absolute_text_proto_path = sight_repo_path.joinpath(
        question_config['attrs_text_proto'])
    absolute_text_proto_path = sight_repo_path.joinpath(
        question_config['attrs_text_proto'])

    if not os.path.exists(absolute_text_proto_path):
      raise FileNotFoundError(f'File not found {relative_text_proto_path}')
    if not os.path.exists(absolute_text_proto_path):
      raise FileNotFoundError(f'File not found {relative_text_proto_path}')

    with open(absolute_text_proto_path, 'r') as f:
      text_proto_data = f.read()
    with open(absolute_text_proto_path, 'r') as f:
      text_proto_data = f.read()

  # # Extract attributes
  # action_attrs = decision_helper.config_to_attr(question_config, 'action')
  # state_attrs = decision_helper.config_to_attr(question_config, 'state')
  # outcome_attrs = decision_helper.config_to_attr(question_config, 'outcome')

  # sight.widget_decision_state[
  #     'decision_episode_fn'] = decision_episode_fn.DecisionEpisodeFn(
  #         state_attrs, action_attrs)
  sight.widget_decision_state['num_decision_points'] = 0

  # Set up the decision configuration
  decision_configuration = sight_pb2.DecisionConfigurationStart(
      optimizer_type=opt_obj.optimizer_type(),
      num_trials=optimizer_config['num_questions'],
      question_label=question_label)
  decision_configuration.choice_config[sight.params.label].CopyFrom(
      opt_obj.create_config())

  decision_configuration.server_queue_batch_size = FLAGS.server_queue_batch_size or 1

  Merge(text_proto_data, decision_configuration)

  # decision_helper.attr_dict_to_proto(state_attrs,
  #                                    decision_configuration.state_attrs)
  # decision_helper.attr_dict_to_proto(action_attrs,
  #                                    decision_configuration.action_attrs)
  # decision_helper.attr_dict_to_proto(outcome_attrs,
  #                                    decision_configuration.outcome_attrs)

  # Enter and exit decision block
  sight.enter_block(
      'Decision Configuration',
      sight_pb2.Object(block_start=sight_pb2.BlockStart(
          sub_type=sight_pb2.BlockStart.ST_CONFIGURATION,
          configuration=sight_pb2.ConfigurationStart(
              sub_type=sight_pb2.ConfigurationStart.ST_DECISION_CONFIGURATION,
              decision_configuration=decision_configuration,
          ))))
  sight.exit_block('Decision Configuration', sight_pb2.Object())
  return decision_configuration


def configure_decision(sight, question_label, question_config, optimizer_config,
                       opt_obj):
  # Create optimizer and configure decision state
  decision_configuration = get_decision_configuration_for_opt(
      sight, question_label, opt_obj, question_config, optimizer_config)
  return decision_configuration


def setup_optimizer(
    sight,
    optimizer_type,
    description='') -> Union[LLMOptimizerClient, SingleActionOptimizerClient]:
  """Sets up the optimizer based on the given type.

  Args:
    sight: The Sight object to be used for logging.
    optimizer_type : optimizer_type
    description: Human-readable description of the application.

  Returns:
    The optimizer to be used for training.

  Raises:
    ValueError:
      If the optimizer type is unknown.
  """
  optimizer_map = {
      # 'dm_acme': lambda: AcmeOptimizerClient(sight),
      'vizier':
          lambda: SingleActionOptimizerClient(
              sight_pb2.DecisionConfigurationStart.OptimizerType.OT_VIZIER,
              sight),
      # 'genetic_algorithm': lambda: GeneticAlgorithmOptimizerClient(
      #     max_population_size=_NUM_TRAIN_WORKERS.value, sight=sight),
      'exhaustive_search':
          lambda: SingleActionOptimizerClient(
              sight_pb2.DecisionConfigurationStart.OptimizerType.
              OT_EXHAUSTIVE_SEARCH,
              sight,
          ),
      'bayesian_opt':
          lambda: SingleActionOptimizerClient(
              sight_pb2.DecisionConfigurationStart.OptimizerType.
              OT_BAYESIAN_OPT,
              sight,
          ),
      'sensitivity_analysis':
          lambda: SingleActionOptimizerClient(
              sight_pb2.DecisionConfigurationStart.OptimizerType.
              OT_SENSITIVITY_ANALYSIS,
              sight,
          ),
      'smcpy':
          lambda: SingleActionOptimizerClient(
              sight_pb2.DecisionConfigurationStart.OptimizerType.OT_SMC_PY,
              sight),
      'worklist_scheduler':
          lambda: SingleActionOptimizerClient(
              sight_pb2.DecisionConfigurationStart.OptimizerType.
              OT_WORKLIST_SCHEDULER,
              sight,
          ),
  }

  # Add support for dynamic optimizers
  if optimizer_type.startswith('llm_'):
    return LLMOptimizerClient(
        optimizer_type.partition('llm_')[2], description, sight)

  if optimizer_type.startswith('ng_'):
    return SingleActionOptimizerClient(
        sight_pb2.DecisionConfigurationStart.OptimizerType.OT_NEVER_GRAD,
        sight,
        optimizer_type.partition('ng_')[2],
    )

  if optimizer_type not in optimizer_map:
    raise ValueError(f'Unknown optimizer type {optimizer_type}')

  return optimizer_map[optimizer_type]()


# def initialize_env(env, state_attrs, action_attrs):
#   if env is not None:
#     if not state_attrs:
#       state_attrs.update(attr_to_dict(env.observation_spec(), 'state'))
#     if not action_attrs:
#       action_attrs.update(attr_to_dict(env.action_spec(), 'action'))


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


# Works in case of decision outcome
def get_decision_outcome_proto(outcome_label: str,
                               sight: Any) -> sight_pb2.DecisionOutcome:
  """Returns the decision outcome proto for the given outcome label."""
  decision_outcome_proto = sight_pb2.DecisionOutcome(
      outcome_label=outcome_label)
  if 'sum_reward' in sight.widget_decision_state:
    decision_outcome_proto.reward = sight.widget_decision_state['sum_reward']

  if 'sum_outcome' in sight.widget_decision_state:
    decision_outcome_proto.outcome_params.CopyFrom(
        convert_dict_to_proto(dict=sight.widget_decision_state['sum_outcome']))

  if 'discount' in sight.widget_decision_state:
    decision_outcome_proto.discount = sight.widget_decision_state['discount']

  return decision_outcome_proto


#! need to refactor this to merge it with get_decision_outcome_proto
def get_decision_outcome_from_decision_message(
    outcome_label: str, decision_message: DecisionMessage):
  """Returns the decision outcome from the decision message."""

  logging.info('decision message =>%s', decision_message)

  decision_outcome_proto = sight_pb2.DecisionOutcome(
      outcome_label=outcome_label)
  decision_outcome_proto.reward = decision_message.reward
  decision_outcome_proto.outcome_params.CopyFrom(
      convert_dict_to_proto(dict=decision_message.outcome_params))
  decision_outcome_proto.discount = decision_message.discount
  logging.info('decision decision_outcome_proto =>%s', decision_outcome_proto)
  return decision_outcome_proto


def _configure_client_and_worker(sight):
  """Configures the client and worker identifiers."""
  if FLAGS.worker_mode is None or _TRAINED_MODEL_LOG_ID.value:
    client_id = str(sight.id)
    worker_location = '0'
  elif FLAGS.worker_mode:
    client_id = os.environ['PARENT_LOG_ID']
    worker_location = os.environ['worker_location']
  else:
    client_id = 'unknown'
    worker_location = 'unknown'
  return client_id, worker_location


def _process_acme_action(selected_action, widget_state):
  """Processes the action for 'dm_acme' optimizer."""
  # ? when action space is scalar (DQN agent - cartpole)
  if selected_action.shape == ():
    return {
        widget_state['decision_episode_fn'].action_attrs[0]: selected_action[()]
    }
  # ? when action space is 1d array (D4pg agent - pendulum)
  return {
      widget_state['decision_episode_fn'].action_attrs[i]: selected_action[i]
      for i in range(len(widget_state['decision_episode_fn'].action_attrs))
  }


def _process_cached_messages_scheduler(sight, req):
  """Processes the action for optimizer where we have messages cached while doing
     working-alive call
  """
  widget_state = sight.widget_decision_state
  logging.info(
      '_process_cached_messages_scheduler: optimizer.obj=%s, action_id=%s',
      optimizer.obj, widget_state['action_id'])

  # we have action_id means we already cached them from the workeralive call
  # and not performing the actual server decision call
  if widget_state['action_id']:
    return (widget_state['cached_messages'].get(
        widget_state['action_id']).action_params)
  return optimizer.get_instance().decision_point(sight, req)


def _process_llm_action(sight, req, optimizer_obj):
  """Processes the action for 'llm_' optimizers."""
  widget_state = sight.widget_decision_state
  if 'reward' in widget_state:
    req.decision_outcome.reward = widget_state['reward']
  if 'outcome_value' in widget_state:
    req.decision.outcome.outcome_params.CopyFrom(
        convert_dict_to_proto(dict=widget_state['outcome_value']))
  req.decision_outcome.discount = widget_state['discount']
  return optimizer_obj.decision_point(sight, req)


def _make_decision(sight, req):
  """Handles decision-making based on the optimizer type."""
  optimizer_obj = optimizer.get_instance()
  optimizer_type = _OPTIMIZER_TYPE.value
  widget_state = sight.widget_decision_state
  if optimizer_type == 'dm_acme':
    selected_action = optimizer_obj.decision_point(sight, req)
    chosen_action = _process_acme_action(selected_action, widget_state)
  elif optimizer_type in [
      'vizier',
      'genetic_algorithm',
      'exhaustive_search',
      'smcpy',
  ]:
    # or optimizer_type.startswith('ng_'):
    chosen_action = optimizer_obj.decision_point(sight, req)
  elif optimizer_type in [
      'worklist_scheduler', 'sensitivity_analysis', 'bayesian_opt'
  ] or optimizer_type.startswith('ng_'):
    chosen_action = _process_cached_messages_scheduler(sight, req)
  elif optimizer_type.startswith('llm_'):
    chosen_action = _process_llm_action(sight, req, optimizer_obj)
  else:
    raise ValueError(f'Unsupported optimizer type: {optimizer_type}')
  return chosen_action


def _log_decision(choice_label, chosen_action, sight):
  """Logs the decision to the Sight logger."""
  choice_params = sight_pb2.DecisionParam()
  choice_params.CopyFrom(convert_dict_to_proto(dict=chosen_action))
  obj = sight_pb2.Object(
      sub_type=sight_pb2.Object.ST_DECISION_POINT,
      decision_point=sight_pb2.DecisionPoint(choice_label=choice_label),
  )
  obj.decision_point.choice_params.CopyFrom(choice_params)
  sight.log_object(obj, inspect.currentframe().f_back.f_back)


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
  # logging.info(
  #     '>>>>>>>>>  In %s of %s, sight.widget_decision_state=%s',
  #     method_name,
  #     _file_name,
  #     sight.widget_decision_state,
  # )

  # Increment decision point count

  print(f'sight widget decision state => {sight.widget_decision_state}')

  sight.widget_decision_state['num_decision_points'] += 1

  # Return cached action if available
  if 'constant_action' in sight.widget_decision_state:
    return sight.widget_decision_state['constant_action']

  # Prepare the request
  req = service_pb2.DecisionPointRequest()
  client_id, worker_location = _configure_client_and_worker(sight)
  req.client_id = client_id
  req.worker_id = f'client_{client_id}_worker_{worker_location}'
  req.question_label = choice_label

  # perform the decision-making process
  chosen_action = _make_decision(sight, req)

  # setting the constant_action in sight widget
  sight.widget_decision_state['constant_action'] = chosen_action

  # log the decision
  _log_decision(choice_label, chosen_action, sight)

  logging.info('decision_point() chosen_action=%s', chosen_action)
  logging.debug('<<<< Out %s of %s', method_name, _file_name)
  return chosen_action


def _update_cached_batch(sight: Any):
  """Updates the cached batch with the latest decision state.

  Args:
      sight: Instance of a Sight logger.
  """
  action_id = sight.widget_decision_state.get('action_id', None)
  logging.info('_update_cached_batch() action_id=%s', action_id)
  cached_messages = sight.widget_decision_state.get('cached_messages', None)
  logging.info('_update_cached_batch() cached_messages=%s', cached_messages)
  if cached_messages and action_id:
    logging.info(
        f'_update_cached_batch() Caching batch for action_id: {action_id}')
    cached_messages.update(
        key=action_id,
        action_params=cached_messages.get(action_id).action_params,
        discount=sight.widget_decision_state['discount'],
        reward=sight.widget_decision_state.get('sum_reward', 0),
        outcome_params=sight.widget_decision_state.get('sum_outcome', {}),
    )


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
    reward: The numeric value of the quality of this outcome, with higher values
      being more desirable.
    outcome: Dictionary that describes the various outcome attributes of the
      application.
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
      if utils.is_numeric(outcome[key]):
        if key not in sight.widget_decision_state['sum_outcome']:
          sight.widget_decision_state['sum_outcome'][key] = 0
        sight.widget_decision_state['sum_outcome'][key] += outcome[key]
      # converting json into string
      else:
        # converting pandas datafram to json and storing it as json string
        # sight.widget_decision_state['sum_outcome'][key] =
        # json.dumps(outcome[key].to_json())
        sight.widget_decision_state['sum_outcome'][key] = outcome[key]

  sight.log_object(
      sight_pb2.Object(
          sub_type=sight_pb2.Object.ST_DECISION_OUTCOME,
          decision_outcome=get_decision_outcome_proto(outcome_label, sight),
      ),
      inspect.currentframe().f_back.f_back,
  )

  _update_cached_batch(sight)

  if 'sum_reward' in sight.widget_decision_state:
    _rewards.append(sight.widget_decision_state['sum_reward'])

  sight.widget_decision_state.pop('sum_reward', None)
  sight.widget_decision_state.pop('sum_outcome', None)

  logging.debug('<<<<  Out %s of %s', method_name, _file_name)


def propose_actions(sight, question_label, action_dict):
  """Proposes actions to the server."""

  attr_dict = sight.fetch_attributes()

  request = service_pb2.ProposeActionRequest()
  request.client_id = str(sight.id)
  request.question_label = question_label
  request.action_attrs.CopyFrom(convert_dict_to_proto(dict=action_dict))
  request.attributes.CopyFrom(convert_dict_to_proto(dict=attr_dict))

  response = service.call(
      lambda s, meta: s.ProposeAction(request, 300, metadata=meta))
  action_id = response.action_id

  # log_object call
  # sight_obj = sight_pb2.Object()
  # sight_obj.sub_type = sight_pb2.Object.SubType.ST_PROPOSE_ACTION
  # sight_obj.propose_action.action_id = str(action_id)
  # sight_obj.propose_action.action_attrs.CopyFrom(request.action_attrs)
  # sight_obj.propose_action.attributes.CopyFrom(request.attributes)

  # frame = inspect.currentframe().f_back.f_back
  # sight.set_object_code_loc(sight_obj, frame)
  # sight.log_object(sight_obj, True)

  return action_id


def _handle_optimizer_finalize(sight: Any, req: Any) -> None:
  """Handles optimizer-specific finalization logic.

  Args:
      sight: Instance of a Sight logger.
      req: FinalizeEpisodeRequest object.
  """
  optimizer_obj = optimizer.get_instance()

  # Get the list of action messages (supports multiple action IDs)
  cached_messages_obj = sight.widget_decision_state.get('cached_messages', {})
  logging.info('cached_messages_obj=%s', cached_messages_obj)
  all_messages: dict[str, DecisionMessage] = cached_messages_obj.all_messages()
  logging.info('all_messages => %s', all_messages)

  for action_id, msg in all_messages.items():
    logging.info('action_id=%s, msg=%s', action_id, msg)
    logging.info('msg.action_params=%s', msg.action_params)
    decision_message = sight_pb2.DecisionMessage()
    decision_message.decision_outcome.CopyFrom(
        get_decision_outcome_from_decision_message(outcome_label='outcome',
                                                   decision_message=msg))
    decision_message.action_id = action_id

    choice_params = sight_pb2.DecisionParam()
    choice_params.CopyFrom(convert_dict_to_proto(dict=msg.action_params))
    decision_message.decision_point.choice_params.CopyFrom(choice_params)

    # logging.info('decision_message=%s', decision_message)
    req.decision_messages.append(decision_message)
  logging.info('Finalize req=%s', req)

  # clearing the cached
  cached_messages_obj.clear()

  if _OPTIMIZER_TYPE.value in {
      'genetic_algorithm',
      'exhaustive_search',
      'vizier',
      'bayesian_opt',
      'sensitivity_analysis',
      'smcpy',
  } or _OPTIMIZER_TYPE.value.startswith(('llm_', 'ng_')):
    optimizer_obj.finalize_episode(sight, req)

  elif _OPTIMIZER_TYPE.value == 'worklist_scheduler':
    if not optimizer.obj:
      optimizer.obj = SingleActionOptimizerClient(
          sight_pb2.DecisionConfigurationStart.OptimizerType.
          OT_WORKLIST_SCHEDULER,
          sight,
      )
    optimizer_obj.finalize_episode(sight, req)

  elif _OPTIMIZER_TYPE.value == 'dm_acme':
    optimizer_obj.finalize_episode(sight)

  if 'outcome_value' in sight.widget_decision_state:
    del sight.widget_decision_state['outcome_value']


def finalize_episode(question_label, sight):  # , optimizer_obj
  """Finalize the run.

  Args:
    sight: Instance of a Sight logger.
  """
  method_name = 'finalize_episode'
  logging.debug('>>>>>>>>>  In %s of %s', method_name, _file_name)

  # if FLAGS.server_mode in {'local', 'worker_mode'}:
  client_id, worker_location = _configure_client_and_worker(sight)

  # create the req
  req = service_pb2.FinalizeEpisodeRequest(
      client_id=client_id,
      worker_id=f'client_{client_id}_worker_{worker_location}',
      question_label=question_label)

  _handle_optimizer_finalize(sight, req)

  #! Not sure about this condition so commented as of now
  # else:
  #   logging.info('Not in local/worker mode, so skipping it')

  #   client_id, worker_location = _configure_client_and_worker(sight)

  #   if sight.widget_decision_state['proposed_actions']:
  #     for proposal in sight.widget_decision_state['proposed_actions']:
  #       proposal_req = service_pb2.ProposeActionRequest(
  #           client_id=client_id,
  #           worker_id=f'client_{client_id}_worker_{worker_location}',
  #           outcome=sight_pb2.DecisionOutcome(
  #               outcome_label='estimated_outcome',
  #               outcome_value=proposal['outcome'],
  #           ),
  #           action=proposal['action'],
  #       )
  #       # logging.info('proposal=%s', proposal)
  #       response = service.call(
  #           lambda s, meta: s.ProposeAction(proposal_req, 300, metadata=meta))
  #     sight.widget_decision_state['proposed_actions'] = []

  logging.debug('<<<<  Out %s of %s', method_name, _file_name)


def get_outcome(sight):
  """Returns the outcome from the server.

  Args:
    sight: Instance of a Sight logger.

  Returns:
    outcome_list: List of outcomes
  """
  request = service_pb2.GetOutcomeRequest()
  request.client_id = str(sight.id)
  # request.unique_ids.append(3)
  response = service.call(
      lambda s, meta: s.GetOutcome(request, 300, metadata=meta))

  if response.response_str:
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
