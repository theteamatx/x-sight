# In the sight give the list of projects ID with yes/no
# Based on optimizer decision point get next points create a portfolio
# pass to blackbox which will simulate the portfolio and will return pem value

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
import threading
from typing import Sequence

from absl import app
from absl import flags
from fvs_sight.fvs_api import action_attrs
from fvs_sight.fvs_api import outcome_attrs
import numpy as np
import pandas as pd
from sight import data_structures
from sight import service_utils as service
from sight.attribute import Attribute
from sight.block import Block
from sight.proto import sight_pb2
from sight.sight import Sight
from sight.widgets.decision import decision
from sight.widgets.decision import trials
from sight.widgets.decision.single_action_optimizer_client import (
    SingleActionOptimizerClient
)
from sight_service.optimizer_instance import param_proto_to_dict
from sight_service.proto import service_pb2
from sight.demo.resource_lock import RWLockDictWrapper

FLAGS = flags.FLAGS
POLL_LIMIT = 6 # POLL_TIME_INTERVAL th part of second
POLL_TIME_INTERVAL = 10 # seconds
global_outcome_mapping = RWLockDictWrapper()

def get_sight_instance():
  params = sight_pb2.Params(
      label="kokua_experiment",
      bucket_name=f'{os.environ["PROJECT_ID"]}-sight',
  )
  sight_obj = Sight(params)
  return sight_obj


def launch_dummy_optimizer(sight):
  optimizer_object = SingleActionOptimizerClient(
      sight_pb2.DecisionConfigurationStart.OptimizerType.OT_WORKLIST_SCHEDULER,
      sight)
  decision_configuration = sight_pb2.DecisionConfigurationStart()
  decision_configuration.optimizer_type = optimizer_object.optimizer_type()

  decision_configuration.num_trials = FLAGS.num_trials
  decision_configuration.choice_config[sight.params.label].CopyFrom(
      optimizer_object.create_config())
  # decision._attr_dict_to_proto(state_attrs,
  #                              decision_configuration.state_attrs)
  decision._attr_dict_to_proto(action_attrs,
                               decision_configuration.action_attrs)
  decision._attr_dict_to_proto(outcome_attrs,
                               decision_configuration.outcome_attrs)
  trials.launch(
      optimizer_object,
      decision_configuration,
      FLAGS.num_train_workers,
      sight,
  )

async def push_message(sight_id, action_id):
  try:
    global_outcome_mapping.set_for_key(action_id,None)
  except Exception as e:
    print(f'Exception => {e}')
    raise e

async def fetch_outcome(sight_id, actions_id):
  while True:
    try:
      outcome = global_outcome_mapping.get_for_key(actions_id)
      if outcome:
        return outcome
      else:
        async_dict = global_outcome_mapping.get()
        print(f'GLOBAL_MAPPING_GET_OUTCOME_QUEUE => {async_dict}')
        time = 5
        print(f'Waiting for {actions_id} for {time} seconds...')
        await asyncio.sleep(time)
    except Exception as e:
      raise e

async def propose_actions(sight, action_dict):

  with Attribute('Managed', '0', sight):
    unique_action_id1 = decision.propose_actions(sight, action_dict)
  with Attribute('Managed', '1', sight):
    unique_action_id2 = decision.propose_actions(sight, action_dict)

  # push messsage into QUEUE
  await push_message(sight.id, unique_action_id1)
  await push_message(sight.id, unique_action_id2)

  task1 = asyncio.create_task(fetch_outcome(sight.id, unique_action_id1))
  task2 = asyncio.create_task(fetch_outcome(sight.id, unique_action_id2))

  # wait till we get outcome of all the samples
  time_series = await asyncio.gather(task1, task2)
  print("time_series :", time_series)
  # calculate diff series
  # appy watermark algorithm
  # return the final series
  return time_series

async def main(sight: Sight, argv: Sequence[str]) -> None:
  if len(argv) > 1:
    raise app.UsageError("Too many command-line arguments.")

  sample_list = [{
    "base-FERTILIZ-howManyCycle": 1.0,
    "base-FERTILIZ-extra_step": 0.0,
    "base-FERTILIZ-extra_offset": 0.0
  },
    {
    "base-FERTILIZ-howManyCycle": 0.5,
    "base-FERTILIZ-extra_step": 0.0,
    "base-FERTILIZ-extra_offset": 0.0
  }]

  # print('SIGHT ID => ',sight.id)
  with Block("Propose actions", sight):
    with Attribute("project_id", "APR107", sight):
      tasks = []
      for id in range(len(sample_list)):
        with Attribute("sample_id", id, sight):
          tasks.append(
              asyncio.create_task(propose_actions(sight, sample_list[id])))

      print("waiting for all get outcome to finish.....")
      diff_time_series = await asyncio.gather(*tasks)
      print(f'Combine Series {diff_time_series}')

def start_workers(sight:Sight):
    trials.start_jobs(
        num_train_workers=4,
        num_trials=4,
        binary_path='fvs_sight/fvs_worker.py',
        optimizer_type='worklist_scheduler',
        docker_image='gcr.io/cameltrain/kokua_portfolio:fvs',
        decision_mode='train',
        deployment_mode='worker_mode',
        worker_mode='dsub_cloud_worker',
        sight=sight,
    )

def main_wrapper(argv):
  with get_sight_instance() as sight:
    launch_dummy_optimizer(sight)
    start_workers(sight)
    print('going to sleep for 6 minutes')
    time.sleep(360)

    sight.init_sight_polling_thread()
    asyncio.run(main(sight, argv))

if __name__ == "__main__":
  app.run(main_wrapper)
