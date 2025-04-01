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
"""utility functions to be used in other functionalities."""

import os

from absl import flags
import yaml

FLAGS = flags.FLAGS


def is_scalar(value):
  scalar_types = (int, float, str, bool, type(None), bytes)
  return isinstance(value, scalar_types)


def load_yaml_config(file_path):
  print(f'loading file from {file_path}')
  try:
    with open(file_path, 'r') as f:
      return yaml.safe_load(f)
  except FileNotFoundError:
    print(f"Error: Config file not found at {file_path}")
    exit(1)


def get_worker_version(question_label):
  #? this only works for single type worker attached to each question lable
  #? for multiple workers, need to update the logic
  # current_script_directory = os.path.dirname(os.path.abspath(__file__))

  # !!! NEED TO CORRECT THIS FUNCRION

  return 'v1'

  workers_config_path = os.getenv(
      "WORKERS_CONFIG_PATH")  #, "default/path/to/config.yaml")
  print('workers_config_path from env: ', workers_config_path)
  workers_config = load_yaml_config('/x-sight/' + workers_config_path)
  optimizers_config_path = os.getenv(
      "OPTIMIZERS_CONFIG_PATH")  #, "default/path/to/config.yaml")
  print('optimizers_config_path from env: ', optimizers_config_path)
  optimizers_config = load_yaml_config('/x-sight/' + optimizers_config_path)

  optimizer_config = optimizers_config[question_label]
  # as of now assuming only 1 worker_type for each question
  for worker, worker_count in optimizer_config['workers'].items():
    worker_details = workers_config[worker]
    return worker_details['version']
