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
"""Binary to spawn multiple workers with given file."""

from datetime import datetime
import math
import os
import subprocess
from typing import Any, Callable, Dict, Optional, Sequence, Text, Tuple
from pathlib import Path

from absl import app
from absl import flags
from helpers.logs.logs_handler import logger as logging
from sight.proto import sight_pb2
from sight.sight import Sight
from sight.widgets.decision import utils
from sight.widgets.decision import decision

FLAGS = flags.FLAGS

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


def get_question_label():
  return 'calculator'


def get_sight_instance():
  print('creating sight object')
  params = sight_pb2.Params(
      label='original_demo',
      bucket_name=f'{os.environ["PROJECT_ID"]}-sight',
  )
  sight_obj = Sight(params)
  return sight_obj


def main(argv: Sequence[str]) -> None:
  if len(argv) > 1:
    raise app.UsageError("Too many command-line arguments.")

  current_file = Path(__file__).resolve()
  sight_repo_path = current_file.parents[3]
  absolute_path = str(sight_repo_path) + '/'

  questions_config = utils.load_yaml_config(absolute_path +
                                            _QUESTIONS_CONFIG.value)
  optimizers_config = utils.load_yaml_config(absolute_path +
                                             _OPTIMIZERS_CONFIG.value)
  workers_config = utils.load_yaml_config(absolute_path + _WORKERS_CONFIG.value)

  with get_sight_instance() as sight:

    config_dict = {
        'questions_config': questions_config,
        'optimizers_config': optimizers_config,
        'workers_config': workers_config
    }

    decision.run(sight=sight,
                 question_label=get_question_label(),
                 configs=config_dict)


if __name__ == "__main__":
  app.run(main)
