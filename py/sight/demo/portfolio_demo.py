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
import time
from typing import Any, Sequence
import warnings

from absl import app
from absl import flags
# from fvs_sight.fvs_api import action_attrs
# from fvs_sight.fvs_api import outcome_attrs
from fvs_sight import fvs_api
from helpers.logs.logs_handler import logger as logging
import pandas as pd
from sight.attribute import Attribute
from sight.block import Block
from sight.proto import sight_pb2
from sight.sight import Sight
from sight.widgets.decision import decision
from sight.widgets.decision import proposal
import yaml


def warn(*args, **kwargs):
  pass


warnings.warn = warn

sample = {
    'region': "NC",
}

FLAGS = flags.FLAGS


def get_sight_instance():
  params = sight_pb2.Params(
      label="kokua_experiment",
      bucket_name=f'{os.environ["PROJECT_ID"]}-sight',
  )
  sight_obj = Sight(params)
  return sight_obj


async def propose_actions(sight: Sight, base_project_config: dict[str, Any],
                          treatments: dict[str, Any]) -> pd.Series:
  """Proposes actions for both base and treatment projects.

  Args:
    sight: The Sight instance.
    base_project_config: The configuration for the base project.
    treatments: The configuration for the treatment project.

  Returns:
    A pandas Series containing the proposed actions.
  """

  x_start_time = time.perf_counter()
  logging.info(f"Proposing Start m-um ")

  treatment_project_config = treatments
  tasks = []
  with Attribute("Managed", "0", sight):
    unmanaged_task = sight.create_task(
        proposal.propose_actions(sight, action_dict=base_project_config))
    tasks.append(unmanaged_task)
  with Attribute("Managed", "1", sight):
    managed_task = sight.create_task(
        proposal.propose_actions(sight, action_dict=treatment_project_config))
    tasks.append(managed_task)

  [unmanaged_response, managed_response] = await asyncio.gather(*tasks)

  x_end_time = time.perf_counter()
  logging.info(
      f"Propose actions m-um took {x_end_time - x_start_time:.4f} seconds.")
  return unmanaged_response, managed_response


async def main(sight: Sight, argv: Sequence[str]) -> None:
  if len(argv) > 1:
    raise app.UsageError("Too many command-line arguments.")

  sample_list = [sample for i in range(FLAGS.num_trials)]

  # print('SIGHT ID => ',sight.id)
  with Block("Propose actions", sight):
    with Attribute("project_id", "APR107", sight):
      tasks = []
      logging.info("len(sample_list) : %s", len(sample_list))

      x_start_time = time.perf_counter()
      logging.info(f"Proposing Start ")

      for id in range(len(sample_list)):
        with Attribute("sample_id", id, sight):
          tasks.append(
              sight.create_task(
                  # both base and treatment are considerred to be same dict here
                  propose_actions(sight, sample_list[id], sample_list[id])))

      x_end_time = time.perf_counter()
      logging.info(
          f"Propose actions took {x_end_time - x_start_time:.4f} seconds.")

      logging.info("waiting for all get outcome to finish.....")
      diff_time_series = await asyncio.gather(*tasks)

      logging.info("all get outcome are finished.....")
      logging.info(f'Combine Series : {diff_time_series}')


def main_wrapper(argv):
  with get_sight_instance() as sight:
    # decision.run(action_attrs=fvs_api.get_action_attrs(),
    #              outcome_attrs=fvs_api.get_outcome_attrs(),
    #              sight=sight)
    questions_config = utils.load_yaml_config('fvs_sight/question_config.yaml')
    optimizers_config = utils.load_yaml_config(
        'fvs_sight/optimizer_config.yaml')
    workers_config = utils.load_yaml_config('fvs_sight/worker_config.yaml')

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

    start_time = time.perf_counter()
    sleep_time_in_min = 5
    logging.info(
        f"Waiting for {sleep_time_in_min} min for workers to start ...")
    time.sleep(sleep_time_in_min * 60)
    asyncio.run(main(sight, argv))
    end_time = time.perf_counter()
    elapsed_time = end_time - start_time
    logging.info(f"Elapsed time: {elapsed_time} seconds")
    hours, remainder = divmod(elapsed_time, 3600)
    minutes, seconds = divmod(remainder, 60)

    if hours > 0:
      logging.info(
          f"Elapsed time: {int(hours)} hour(s), {int(minutes)} minute(s),"
          f" {seconds:.2f} second(s)")
    elif minutes > 0:
      logging.info(
          f"Elapsed time: {int(minutes)} minute(s), {seconds:.2f} second(s)")
    else:
      logging.info(f"Elapsed time: {seconds:.2f} second(s)")


if __name__ == "__main__":
  app.run(main_wrapper)
