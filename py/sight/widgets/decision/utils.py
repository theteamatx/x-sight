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

from typing import Any
import os
from pathlib import Path
from absl import flags
import yaml

FLAGS = flags.FLAGS

def get_config_dir_path() -> str:
  current_file = Path(__file__).resolve()
  root_repo_path = find_root_repo(current_file)
  #todo tmp-fix - works for kokua demo, need to change this dynamically
  config_dir_path = str(root_repo_path) + '/analytics/optimizer/config'
  return config_dir_path


def is_numeric(val: Any) -> bool:
  return isinstance(val, (int, float))


def load_yaml_config(file_path: str) -> str:
  print(f'loading file from {file_path}')
  try:
    with open(file_path, 'r') as f:
      return yaml.safe_load(f)
  except FileNotFoundError:
    print(f"Error: Config file not found at {file_path}")
    exit(1)

def find_root_repo(current_path):
    # root directory in docker image is fixed - can't find it based on .git folder
    if os.path.exists('/.dockerenv'):
        return Path('/x-sight')
    else:
      current_dir = Path(current_path).resolve()
      while current_dir != current_dir.parent:
          git_path = current_dir / '.git'
          if git_path.is_dir():  # Only checks for a directory to eliminate submodule
              return current_dir
          current_dir = current_dir.parent
      raise ValueError(f'No root folder found...')

def get_worker_version(question_label:str, sight)-> str:
  # !!! NEED TO CORRECT THIS FUNCRION
  return 'v1'

  optimizer_config = sight.get_decision_config().optimizers[question_label]
  # as of now assuming only 1 worker_type for each question
  for worker in sorted(optimizer_config['workers'].keys()):
    worker_details = sight.get_decision_config().workers[worker]
    return worker_details['version']
  raise ValueError(f'No configuration for question label {question_label}')

