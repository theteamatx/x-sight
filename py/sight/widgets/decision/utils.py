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
from pathlib import Path
from absl import flags
import yaml

FLAGS = flags.FLAGS

def get_config_dir_path():
  current_file = Path(__file__).resolve()
  sight_repo_path = current_file.parents[4]
  config_dir_path = str(sight_repo_path) + '/py/sight/configs'
  return config_dir_path


def is_numeric(val):
  return isinstance(val, (int, float))


def load_yaml_config(file_path):
  print(f'loading file from {file_path}')
  try:
    with open(file_path, 'r') as f:
      return yaml.safe_load(f)
  except FileNotFoundError:
    print(f"Error: Config file not found at {file_path}")
    exit(1)


def get_worker_version(question_label, sight):
  worker_name=list(sight.get_decision_config().optimizers[question_label]['workers'].keys())[0]
  return sight.get_decision_config().workers[worker_name]['version']
