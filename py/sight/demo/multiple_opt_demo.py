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
"""Demo of spawning multiple worker which can interact with each other."""

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
from helpers.logs.logs_handler import logger as logging
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

def main(argv: Sequence[str]) -> None:
  if len(argv) > 1:
    raise app.UsageError("Too many command-line arguments.")

  # config contains the data from all the config files
  config = decision.DecisionConfig(config_dir_path=FLAGS.config_path)

  # create sight object with configuration to spawn workers beforehand
  with Sight.create('multiple_opt_label', config) as sight:
    logging.info("spawned the workers.................")


if __name__ == "__main__":
  app.run(main)
