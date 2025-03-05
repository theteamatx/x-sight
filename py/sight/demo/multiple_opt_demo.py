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
"""Demo of using the Sight Decision API to run forest simulator."""

import time
import warnings


def warn(*args, **kwargs):
  pass


warnings.warn = warn

import asyncio
import os
from typing import Any, Sequence

from absl import app
from absl import flags
import pandas as pd
from sight import utility
from sight.attribute import Attribute
from sight.block import Block
from sight.proto import sight_pb2
from sight.sight import Sight
from sight.widgets.decision import decision
from sight.widgets.decision import decision_episode_fn
from sight.widgets.decision import decision_helper
from sight.widgets.decision import proposal
from sight.widgets.decision import trials
from sight.widgets.decision import utils
import yaml

_QUESTIONS_CONFIG = flags.DEFINE_string(
    'questions_config_path',
    'fvs_sight/question_config.yaml',
    'Path of config.yaml containing question related info.',
)

_OPTIMIZERS_CONFIG = flags.DEFINE_string(
    'optimizers_config_path',
    'fvs_sight/optimizer_config.yaml',
    'Path of config.yaml containing optimizer related info.',
)

_WORKERS_CONFIG = flags.DEFINE_string(
    'workers_config_path',
    'fvs_sight/worker_config.yaml',
    'Path of config.yaml containing worker related info.',
)

FLAGS = flags.FLAGS

sample = {'project_id': '133a6365-01cf-4b5e-8197-d4779e5ce25c', 'region': 'NC'}


def get_sight_instance():
  params = sight_pb2.Params(
      label="kokua_experiment",
      bucket_name=f'{os.environ["PROJECT_ID"]}-sight',
  )
  sight_obj = Sight(params)
  return sight_obj


async def propose_actions(sight: Sight, question_label: str,
                          base_project_config: dict[str, Any],
                          treatments: dict[str, Any]) -> pd.Series:
  treatment_project_config = treatments
  tasks = []
  with Attribute("Managed", "0", sight):
    # base_sim = decision.propose_actions(sight,
    #                                       action_dict=base_project_config)
    # await proposal.push_message(sight.id, base_sim)
    # unmanaged_task = sight.create_task(
    #     proposal.fetch_outcome(sight.id, base_sim))
    # tasks.append(unmanaged_task)
    unmanaged_task = sight.create_task(
        proposal.propose_actions(sight,
                                 question_label,
                                 action_dict=base_project_config))
    tasks.append(unmanaged_task)
  with Attribute("Managed", "1", sight):
    # treatment_sim = decision.propose_actions(
    #     sight, action_dict=treatment_project_config)
    # await proposal.push_message(sight.id, treatment_sim)
    # managed_task = sight.create_task(
    #     proposal.fetch_outcome(sight.id, treatment_sim))
    # tasks.append(managed_task)
    managed_task = sight.create_task(
        proposal.propose_actions(sight,
                                 question_label,
                                 action_dict=treatment_project_config))
    tasks.append(managed_task)

  [unmanaged_response, managed_response] = await asyncio.gather(*tasks)
  return unmanaged_response, managed_response


async def propose_actions_wrapper(sight: Sight, question_label: str,
                                  num_trials: int) -> None:

  sample_list = [sample for i in range(num_trials)]

  # print('SIGHT ID => ',sight.id)
  with Block("Propose actions", sight):
    with Attribute("project_id", "APR107", sight):
      tasks = []
      print("len(sample_list) : ", len(sample_list))
      for id in range(len(sample_list)):
        await asyncio.sleep(0.01)
        with Attribute("sample_id", id, sight):
          tasks.append(
              sight.create_task(
                  # both base and treatment are considerred to be same dict here
                  propose_actions(sight, question_label, sample_list[id],
                                  sample_list[id])))

      print("waiting for all get outcome to finish.....")
      diff_time_series = await asyncio.gather(*tasks)
      print("all get outcome are finished.....")
      print(f'Combine Series : {diff_time_series}')


# def get_decision_configuration_for_opt(sight, question_label, opt_obj,
#                                        question_config, optimizer_config):

#   # Extract attributes
#   action_attrs = decision_helper.config_to_attr(question_config, 'action')
#   state_attrs = decision_helper.config_to_attr(question_config, 'state')
#   outcome_attrs = decision_helper.config_to_attr(question_config, 'outcome')

#   sight.widget_decision_state[
#       'decision_episode_fn'] = decision_episode_fn.DecisionEpisodeFn(
#           state_attrs, action_attrs)
#   sight.widget_decision_state['num_decision_points'] = 0

#   # Set up the decision configuration
#   decision_configuration = sight_pb2.DecisionConfigurationStart(
#       optimizer_type=opt_obj.optimizer_type(),
#       num_trials=optimizer_config['num_questions'],
#       question_label=question_label)
#   decision_configuration.choice_config[sight.params.label].CopyFrom(
#       opt_obj.create_config())

#   decision_configuration.server_queue_batch_size = FLAGS.server_queue_batch_size or 1
#   decision_helper.attr_dict_to_proto(state_attrs,
#                                      decision_configuration.state_attrs)
#   decision_helper.attr_dict_to_proto(action_attrs,
#                                      decision_configuration.action_attrs)
#   decision_helper.attr_dict_to_proto(outcome_attrs,
#                                      decision_configuration.outcome_attrs)

#   # Enter and exit decision block
#   sight.enter_block(
#       'Decision Configuration',
#       sight_pb2.Object(block_start=sight_pb2.BlockStart(
#           sub_type=sight_pb2.BlockStart.ST_CONFIGURATION,
#           configuration=sight_pb2.ConfigurationStart(
#               sub_type=sight_pb2.ConfigurationStart.ST_DECISION_CONFIGURATION,
#               decision_configuration=decision_configuration,
#           ))))
#   sight.exit_block('Decision Configuration', sight_pb2.Object())
#   return decision_configuration

# def configure_decision(sight, question_label, question_config, optimizer_config,
#                        optimizer_type):
#   # Create optimizer and configure decision state
#   opt_obj = decision.setup_optimizer(sight, optimizer_type)
#   decision_configuration = get_decision_configuration_for_opt(
#       sight, question_label, opt_obj, question_config, optimizer_config)
#   return opt_obj, decision_configuration


def start_worker_jobs(sight, optimizer_config, worker_configs, optimizer_type):
  # for worker_name in optimizer_config['worker_names']:
  #   worker_details = worker_configs[worker_name]

  num_questions = optimizer_config['num_questions']
  for worker, worker_count in optimizer_config['workers'].items():
    # print('worker_count : ', worker_count)
    worker_details = worker_configs[worker]
    if (optimizer_config['mode'] == 'dsub_cloud_worker'):
      trials.start_jobs(worker_count, worker_details['binary'], optimizer_type,
                        worker_details['docker'], 'train', 'worker_mode',
                        optimizer_config['mode'], sight)
    elif (optimizer_config['mode'] == 'dsub_local_worker'):
      trials.start_job_in_dsub_local(worker_count, worker_details['binary'],
                                     optimizer_type, worker_details['docker'],
                                     'train', 'worker_mode',
                                     optimizer_config['mode'], sight)

    else:
      raise ValueError(
          f"{optimizer_config['mode']} mode from optimizer_config not supported"
      )


def main_wrapper(argv):
  # start_time = time.perf_counter()
  with get_sight_instance() as sight:

    questions_config = utils.load_yaml_config(_QUESTIONS_CONFIG.value)
    optimizers_config = utils.load_yaml_config(_OPTIMIZERS_CONFIG.value)
    workers_config = utils.load_yaml_config(_WORKERS_CONFIG.value)

    # put this in function
    for question_label, question_config in questions_config.items():
      optimizer_type = optimizers_config[question_label]['optimizer']
      optimizer_config = optimizers_config[question_label]
      print('optimizer_config : ', optimizer_config)

      opt_obj = decision.setup_optimizer(sight, optimizer_type)
      decision_configuration = decision.configure_decision(
          sight, question_label, question_config, optimizer_config, opt_obj)
      trials.launch(decision_configuration, sight)

      # Start worker jobs
      start_worker_jobs(sight, optimizer_config, workers_config, optimizer_type)

      # # # propose_action()
      # asyncio.run(
      #     propose_actions_wrapper(sight, question_label,
      #                             optimizer_config['num_questions']))

  # end_time = time.perf_counter()
  # utility.calculate_exp_time(start_time, end_time)


if __name__ == "__main__":
  app.run(main_wrapper)

# Started exp for the 5000*2 tasks and batch-size 100 and workers 50 , lets see

# we have VM server running on left side
# and its docker container stats on right side

# here we are waiting for 8 min , to warn the workers , 8mins means till counter => 220 , here is the
