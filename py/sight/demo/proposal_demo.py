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
"""Demo of using the Sight Propose action API to add actions to server and wait for it's outcome."""
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
from helpers.logs.logs_handler import logger as logging

FLAGS = flags.FLAGS


def get_question_label():
  return 'calculator'

async def propose_actions(
    sight: Sight,
    question_label: str,
    actions: dict[str, Any],
) -> pd.Series:
  tasks = []
  with Attribute("task", "multiply", sight):
    task = sight.create_task(
        proposal.propose_actions(sight, question_label, action_dict=actions))
    tasks.append(task)

  [final_result] = await asyncio.gather(*tasks)
  return final_result


async def propose_actions_wrapper(sight: Sight, question_label: str,
                                  actions: dict) -> None:

  with Block("Propose actions", sight):
    tasks = []
    tasks.append(
        sight.create_task(propose_actions(sight, question_label, actions)))

    logging.info("waiting for all get outcome to finish.....")
    result = await asyncio.gather(*tasks)
    logging.info(f'result : {result}')


def get_sight_instance(config=None):
  params = sight_pb2.Params(
      label=get_question_label(),
      bucket_name=f'{os.environ["PROJECT_ID"]}-sight',
  )
  sight_obj = Sight(params, config)
  return sight_obj


def main(argv: Sequence[str]) -> None:
  if len(argv) > 1:
    raise app.UsageError("Too many command-line arguments.")

  # config contains the data from all the config files
  config = decision.DecisionConfig(config_dir_path=FLAGS.config_path)

  # create sight object with configuration to spawn workers beforehand
  with get_sight_instance(config) as sight:

    # this thread checks the outcome for proposed action from server
    decision.init_sight_polling_thread(sight.id, get_question_label())

    #Ideally this actions will be proposed from some other module
    actions = {"v1": 3, "v2": 5, "ops": 'multiply'}

    asyncio.run(propose_actions_wrapper(sight, get_question_label(), actions))


if __name__ == "__main__":
  app.run(main)
