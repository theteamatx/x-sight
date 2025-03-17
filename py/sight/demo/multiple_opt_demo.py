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


def start_worker_jobs(sight, optimizer_config, worker_configs, optimizer_type):

  num_questions = optimizer_config['num_questions']
  for worker, worker_count in optimizer_config['workers'].items():
    # print('worker_count : ', worker_count)
    worker_file_path = worker_configs[worker]['file_path']
    worker_config = utils.load_yaml_config(worker_file_path)
    worker_details = worker_config[worker_configs[worker]['version']]

    # print('worker_details : ', worker_details)
    # raise SystemExit

    if (optimizer_config['mode'] == 'dsub_cloud_worker'):
      trials.start_jobs(worker_count, worker_details['binary'], optimizer_type,
                        worker_details['docker'], 'train', 'worker_mode',
                        optimizer_config['mode'],
                        FLAGS.cache_mode, sight)
    elif (optimizer_config['mode'] == 'dsub_local_worker'):
      trials.start_job_in_dsub_local(worker_count, worker_details['binary'],
                                     optimizer_type, worker_details['docker'],
                                     'train', 'worker_mode',
                                     optimizer_config['mode'],
                                     FLAGS.cache_mode, sight)

    else:
      raise ValueError(
          f"{optimizer_config['mode']} mode from optimizer_config not supported"
      )


def main_wrapper(argv):
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


if __name__ == "__main__":
  app.run(main_wrapper)
