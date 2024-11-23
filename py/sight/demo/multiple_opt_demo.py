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
import yaml
from typing import Sequence, Any

from absl import app
from absl import flags
import pandas as pd
from sight.attribute import Attribute
from sight.block import Block
from sight.proto import sight_pb2
from sight.sight import Sight
from sight.widgets.decision import decision
from sight.widgets.decision import proposal
from sight import utility
from sight.widgets.decision import decision_helper
from sight.widgets.decision import decision_episode_fn
from sight.widgets.decision import trials
from sight.widgets.decision import utils

FLAGS = flags.FLAGS


def get_sight_instance():
  params = sight_pb2.Params(
      label="kokua_experiment",
      bucket_name=f'{os.environ["PROJECT_ID"]}-sight',
  )
  sight_obj = Sight(params)
  return sight_obj


# async def main(sight: Sight, argv: Sequence[str]) -> None:
#     if len(argv) > 1:
#         raise app.UsageError("Too many command-line arguments.")

#     sample_list = [sample for i in range(FLAGS.num_trials)]

#     # print('SIGHT ID => ',sight.id)
#     with Block("Propose actions", sight):
#         with Attribute("project_id", "APR107", sight):
#             tasks = []
#             print("len(sample_list) : ", len(sample_list))
#             for id in range(len(sample_list)):
#                 with Attribute("sample_id", id, sight):
#                     tasks.append(
#                         sight.create_task(
#                             # both base and treatment are considerred to be same dict here
#                              propose_actions(sight, sample_list[id], sample_list[id])
#                         )
#                     )

#             print("waiting for all get outcome to finish.....")
#             diff_time_series = await asyncio.gather(*tasks)
#             print("all get outcome are finished.....")
#             print(f'Combine Series : {diff_time_series}')

# def generate_attrs_dict(attrs):
#   attr_dict = {}
#   for k, v in attrs.items():
#     attr_dict[k] = sight_pb2.DecisionConfigurationStart.AttrProps(
#         min_value=v[0],
#         max_value=v[1],
#     )
#   return attr_dict


# def load_yaml_config(file_path):
#   with open(file_path, 'r') as f:
#     return yaml.safe_load(f)


def configure_decision(sight, question_label, question_config, optimizer_config,
                       optimizer_type):
  # Extract attributes
  action_attrs = decision_helper.config_to_attr(question_config, 'action')
  state_attrs = decision_helper.config_to_attr(question_config, 'state')
  outcome_attrs = decision_helper.config_to_attr(question_config, 'outcome')

  # Create optimizer and configure decision state
  opt_obj = decision.get_optimizer(optimizer_type, sight)
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

  return opt_obj, decision_configuration


def start_worker_jobs(sight, optimizer_config, worker_configs, optimizer_type):
  # for worker_name in optimizer_config['worker_names']:
  #   worker_details = worker_configs[worker_name]
  for worker, worker_count in optimizer_config['workers'].items():
    worker_details = worker_configs[worker]
    if(optimizer_config['mode'] == 'dsub_cloud_worker'):
      trials.start_jobs(worker_count,
                        worker_details['binary'], optimizer_type,
                        worker_details['docker'], 'train', 'worker_mode',
                        optimizer_config['mode'], sight)
    elif(optimizer_config['mode'] == 'dsub_local_worker'):
      trials.start_job_in_dsub_local(worker_count,
                                     worker_details['binary'], optimizer_type,
                                     worker_details['docker'], 'train', 'worker_mode',
                                     optimizer_config['mode'], sight)

    else:
      raise ValueError(f"{optimizer_config['mode']} mode from optimizer_config not supported")


def main_wrapper(argv):
  # start_time = time.perf_counter()
  with get_sight_instance() as sight:

    question_configs = utils.load_yaml_config('fvs_sight/question_config.yaml')
    optimizer_configs = utils.load_yaml_config('fvs_sight/optimizer_config.yaml')
    worker_configs = utils.load_yaml_config('fvs_sight/worker_config.yaml')

    for question_label, question_config in question_configs.items():
      optimizer_type = optimizer_configs[question_label]['optimizer']
      optimizer_config = optimizer_configs[question_label]

      # Configure decision and launch trials
      opt_obj, decision_configuration = configure_decision(
          sight, question_label, question_config, optimizer_config,
          optimizer_type)
      trials.launch(opt_obj, decision_configuration, sight)

      # Start worker jobs
      start_worker_jobs(sight, optimizer_config, worker_configs, optimizer_type)

  # end_time = time.perf_counter()
  # utility.calculate_exp_time(start_time, end_time)


if __name__ == "__main__":
  app.run(main_wrapper)
