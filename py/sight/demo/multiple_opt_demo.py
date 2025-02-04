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

_QUESTION_CONFIG = flags.DEFINE_string(
    'question_config_path',
    'fvs_sight/question_config.yaml',
    'Path of config.yaml containing question related info.',
)

_OPTIMIZER_CONFIG = flags.DEFINE_string(
    'optimizer_config_path',
    'fvs_sight/optimizer_config.yaml',
    'Path of config.yaml containing optimizer related info.',
)

_WORKER_CONFIG = flags.DEFINE_string(
    'worker_config_path',
    'fvs_sight/worker_config.yaml',
    'Path of config.yaml containing worker related info.',
)

FLAGS = flags.FLAGS

sample = {
    'project_id': '133a6365-01cf-4b5e-8197-d4779e5ce25c',
    'fire-SIMFIRE_0-6_stand_area_burned': 100,
    'fire-SIMFIRE_0-1_cycle': 2013,
    'fire-SIMFIRE_1-6_stand_area_burned': 71,
    'fire-SIMFIRE_1-1_cycle': 2014,
    'fire-SIMFIRE_2-6_stand_area_burned': 100,
    'fire-SIMFIRE_2-1_cycle': 2015,
    'fire-SIMFIRE_4-6_stand_area_burned': 10,
    'fire-SIMFIRE_4-1_cycle': 2017,
    'fire-SIMFIRE_11-6_stand_area_burned': 80,
    'fire-SIMFIRE_11-1_cycle': 2024,
    'fire-SIMFIRE_17-6_stand_area_burned': 45,
    'fire-SIMFIRE_17-1_cycle': 2030,
    'fire-SIMFIRE_19-6_stand_area_burned': 45,
    'fire-SIMFIRE_19-1_cycle': 2032,
    'fire-SIMFIRE_20-6_stand_area_burned': 21,
    'fire-SIMFIRE_20-1_cycle': 2033,
    'fire-SIMFIRE_22-6_stand_area_burned': 34,
    'fire-SIMFIRE_22-1_cycle': 2035,
    'fire-SIMFIRE_23-6_stand_area_burned': 100,
    'fire-SIMFIRE_23-1_cycle': 2036,
    'fire-SIMFIRE_26-6_stand_area_burned': 16,
    'fire-SIMFIRE_26-1_cycle': 2039,
    'fire-SIMFIRE_28-6_stand_area_burned': 100,
    'fire-SIMFIRE_28-1_cycle': 2041,
    'fire-SIMFIRE_29-6_stand_area_burned': 7,
    'fire-SIMFIRE_29-1_cycle': 2042,
    'fire-SIMFIRE_33-6_stand_area_burned': 100,
    'fire-SIMFIRE_33-1_cycle': 2046,
    'fire-SIMFIRE_35-6_stand_area_burned': 87,
    'fire-SIMFIRE_35-1_cycle': 2048,
    'fire-SIMFIRE_36-6_stand_area_burned': 53,
    'fire-SIMFIRE_36-1_cycle': 2049,
    'fire-SIMFIRE_37-6_stand_area_burned': 51,
    'fire-SIMFIRE_37-1_cycle': 2050,
    'fire-SIMFIRE_39-6_stand_area_burned': 8,
    'fire-SIMFIRE_39-1_cycle': 2052,
    'fire-SIMFIRE_42-6_stand_area_burned': 100,
    'fire-SIMFIRE_42-1_cycle': 2055,
    'fire-SIMFIRE_43-6_stand_area_burned': 95,
    'fire-SIMFIRE_43-1_cycle': 2056,
    'fire-SIMFIRE_44-6_stand_area_burned': 14,
    'fire-SIMFIRE_44-1_cycle': 2057,
    'fire-SIMFIRE_45-6_stand_area_burned': 18,
    'fire-SIMFIRE_45-1_cycle': 2058,
    'fire-SIMFIRE_47-6_stand_area_burned': 100,
    'fire-SIMFIRE_47-1_cycle': 2060,
    'fire-SIMFIRE_49-6_stand_area_burned': 100,
    'fire-SIMFIRE_49-1_cycle': 2062,
    'fire-SIMFIRE_50-6_stand_area_burned': 25,
    'fire-SIMFIRE_50-1_cycle': 2063,
    'fire-SIMFIRE_53-6_stand_area_burned': 100,
    'fire-SIMFIRE_53-1_cycle': 2066,
    'fire-SIMFIRE_54-6_stand_area_burned': 66,
    'fire-SIMFIRE_54-1_cycle': 2067,
    'fire-SIMFIRE_56-6_stand_area_burned': 63,
    'fire-SIMFIRE_56-1_cycle': 2069,
    'fire-SIMFIRE_59-6_stand_area_burned': 45,
    'fire-SIMFIRE_59-1_cycle': 2072,
    'fire-SIMFIRE_60-6_stand_area_burned': 79,
    'fire-SIMFIRE_60-1_cycle': 2073,
    'fire-SIMFIRE_63-6_stand_area_burned': 80,
    'fire-SIMFIRE_63-1_cycle': 2076,
    'fire-SIMFIRE_64-6_stand_area_burned': 47,
    'fire-SIMFIRE_64-1_cycle': 2077,
    'fire-SIMFIRE_65-6_stand_area_burned': 64,
    'fire-SIMFIRE_65-1_cycle': 2078,
    'fire-SIMFIRE_66-6_stand_area_burned': 100,
    'fire-SIMFIRE_66-1_cycle': 2079,
    'fire-SIMFIRE_68-6_stand_area_burned': 100,
    'fire-SIMFIRE_68-1_cycle': 2081,
    'fire-SIMFIRE_70-6_stand_area_burned': 30,
    'fire-SIMFIRE_70-1_cycle': 2083,
    'fire-SIMFIRE_71-6_stand_area_burned': 12,
    'fire-SIMFIRE_71-1_cycle': 2084,
    'fire-SIMFIRE_72-6_stand_area_burned': 51,
    'fire-SIMFIRE_72-1_cycle': 2085,
    'fire-SIMFIRE_75-6_stand_area_burned': 17,
    'fire-SIMFIRE_75-1_cycle': 2088,
    'fire-SIMFIRE_76-6_stand_area_burned': 100,
    'fire-SIMFIRE_76-1_cycle': 2089,
    'fire-SIMFIRE_79-6_stand_area_burned': 60,
    'fire-SIMFIRE_79-1_cycle': 2092,
    'fire-SIMFIRE_81-6_stand_area_burned': 45,
    'fire-SIMFIRE_81-1_cycle': 2094,
    'fire-SIMFIRE_84-6_stand_area_burned': 100,
    'fire-SIMFIRE_84-1_cycle': 2097,
    'fire-SIMFIRE_88-6_stand_area_burned': 58,
    'fire-SIMFIRE_88-1_cycle': 2101,
    'fire-SIMFIRE_90-6_stand_area_burned': 82,
    'fire-SIMFIRE_90-1_cycle': 2103,
    'fire-SIMFIRE_92-6_stand_area_burned': 60,
    'fire-SIMFIRE_92-1_cycle': 2105,
    'fire-SIMFIRE_94-6_stand_area_burned': 56,
    'fire-SIMFIRE_94-1_cycle': 2107,
    'fire-SIMFIRE_96-6_stand_area_burned': 100,
    'fire-SIMFIRE_96-1_cycle': 2109,
    'fire-SIMFIRE_97-6_stand_area_burned': 3,
    'fire-SIMFIRE_97-1_cycle': 2110,
    'fire-SIMFIRE_98-6_stand_area_burned': 87,
    'fire-SIMFIRE_98-1_cycle': 2111,
    'region': 'NC',
    'base-FERTILIZ-howManyCycle': 1.0,
    'base-FERTILIZ-extra_step': 0.0,
    'base-FERTILIZ-extra_offset': 0.0
}


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


def get_decision_configuration_for_opt(sight, question_label, opt_obj,
                                       question_config, optimizer_config):

  # Extract attributes
  action_attrs = decision_helper.config_to_attr(question_config, 'action')
  state_attrs = decision_helper.config_to_attr(question_config, 'state')
  outcome_attrs = decision_helper.config_to_attr(question_config, 'outcome')

  sight.widget_decision_state[
      'decision_episode_fn'] = decision_episode_fn.DecisionEpisodeFn(
          state_attrs, action_attrs)
  sight.widget_decision_state['num_decision_points'] = 0

  # Set up the decision configuration
  decision_configuration = sight_pb2.DecisionConfigurationStart(
      optimizer_type=opt_obj.optimizer_type(),
      num_trials=optimizer_config['num_questions'],
      question_label=question_label)
  decision_configuration.choice_config[sight.params.label].CopyFrom(
      opt_obj.create_config())

  decision_configuration.server_queue_batch_size = FLAGS.server_queue_batch_size or 1
  decision_helper.attr_dict_to_proto(state_attrs,
                                     decision_configuration.state_attrs)
  decision_helper.attr_dict_to_proto(action_attrs,
                                     decision_configuration.action_attrs)
  decision_helper.attr_dict_to_proto(outcome_attrs,
                                     decision_configuration.outcome_attrs)

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
                       optimizer_type):
  # Create optimizer and configure decision state
  opt_obj = decision.setup_optimizer(sight, optimizer_type)
  decision_configuration = get_decision_configuration_for_opt(
      sight, question_label, opt_obj, question_config, optimizer_config)
  return opt_obj, decision_configuration


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

    question_configs = utils.load_yaml_config(FLAGS.question_config_path)
    optimizer_configs = utils.load_yaml_config(FLAGS.optimizer_config_path)
    worker_configs = utils.load_yaml_config(FLAGS.worker_config_path)

    for question_label, question_config in question_configs.items():
      optimizer_type = optimizer_configs[question_label]['optimizer']
      optimizer_config = optimizer_configs[question_label]
      print('optimizer_config : ', optimizer_config)

      # Configure decision and launch trials
      opt_obj, decision_configuration = configure_decision(
          sight, question_label, question_config, optimizer_config,
          optimizer_type)
      trials.launch(decision_configuration, sight)

      # Start worker jobs
      start_worker_jobs(sight, optimizer_config, worker_configs, optimizer_type)

      # # propose_action()
      asyncio.run(
          propose_actions_wrapper(sight, question_label,
                                  optimizer_config['num_questions']))

  # end_time = time.perf_counter()
  # utility.calculate_exp_time(start_time, end_time)


if __name__ == "__main__":
  app.run(main_wrapper)
