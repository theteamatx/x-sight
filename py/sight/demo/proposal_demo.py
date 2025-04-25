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
import warnings


def warn(*args, **kwargs):
  pass


warnings.warn = warn

import asyncio
import inspect
import json
import os
import random
from typing import Sequence, Any

from absl import app
from absl import flags
import numpy as np
import pandas as pd
from sight.attribute import Attribute
from sight.block import Block
from sight import data_structures
from sight.proto import sight_pb2
from sight.sight import Sight
from sight.widgets.decision import decision
from sight.widgets.decision import proposal
from sight.widgets.decision import utils
from helpers.logs.logs_handler import logger as logging
FLAGS = flags.FLAGS

def get_question_label():
  return 'Q_label4'

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

async def propose_actions(sight: Sight, question_label: str,
                          actions: dict[str, Any],
                          ) -> pd.Series:
  tasks = []
  with Attribute("task", "multiply", sight):
    task = sight.create_task(
        proposal.propose_actions(sight,
                                 question_label,
                                 action_dict=actions))
    tasks.append(task)

  [final_result] = await asyncio.gather(*tasks)
  return final_result


async def propose_actions_wrapper(sight: Sight, question_label: str) -> None:

  actions = {"v1" : 3, "v2" : 5, "ops" : "multiplication"}
  with Block("Propose actions", sight):
    tasks = []
    tasks.append(
        sight.create_task(
            propose_actions(sight, question_label, actions)))

    logging.info("waiting for all get outcome to finish.....")
    result = await asyncio.gather(*tasks)
    logging.info(f'result : {result}')


def get_sight_instance():
  params = sight_pb2.Params(
      label=get_question_label(),
      bucket_name=f'{os.environ["PROJECT_ID"]}-sight',
  )
  sight_obj = Sight(params)
  return sight_obj


def main(argv: Sequence[str]) -> None:
  if len(argv) > 1:
    raise app.UsageError("Too many command-line arguments.")

  import os
  from pathlib import Path

  current_file = Path(__file__).resolve()
  sight_repo_path = current_file.parents[3]
  absolute_path = str(sight_repo_path) + '/'

  questions_config = utils.load_yaml_config(absolute_path +
                                            _QUESTIONS_CONFIG.value)
  optimizers_config = utils.load_yaml_config(absolute_path +
                                              _OPTIMIZERS_CONFIG.value)
  workers_config = utils.load_yaml_config(absolute_path +
                                          _WORKERS_CONFIG.value)

  with get_sight_instance() as sight:

    config_dict = {
          'questions_config': questions_config,
          'optimizers_config': optimizers_config,
          'workers_config': workers_config
      }

    decision.run(
        sight=sight,
        question_label=get_question_label(),
        configs=config_dict
    )


    # this thread checks the outcome for proposed action from server
    decision.init_sight_polling_thread(sight.id,
                                       get_question_label())
    asyncio.run(
        propose_actions_wrapper(sight, get_question_label()))



if __name__ == "__main__":
  app.run(main)
