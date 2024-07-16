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

import os
from typing import Sequence
import asyncio

from absl import app
from absl import flags
from sight import data_structures
from sight.proto import sight_pb2
from sight.sight import Sight
from sight.attribute import Attribute
from sight.block import Block
from sight.widgets.decision import decision
import pandas as pd
from sight.widgets.decision.single_action_optimizer_client import SingleActionOptimizerClient
from sight.widgets.decision import trials
from fvs_sight.fvs_api import action_attrs, outcome_attrs
from sight_service.proto import service_pb2
from sight import service_utils as service
from sight_service.optimizer_instance import param_proto_to_dict
FLAGS = flags.FLAGS

def get_sight_instance():
    params = sight_pb2.Params(
        label="kokua_experiment",
        bucket_name=f'{os.environ["PROJECT_ID"]}-sight',
    )
    sight_obj = Sight(params)
    return sight_obj


def launch_dummy_optimizer(sight):
    optimizer_object = SingleActionOptimizerClient(
        sight_pb2.DecisionConfigurationStart.OptimizerType.OT_WORKLIST_SCHEDULER, sight)
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

async def get_outcome(sight_id, action_id):

    request = service_pb2.GetOutcomeRequest()
    request.client_id = str(sight_id)
    request.unique_ids.append(action_id)

    # print(f'awaiting for 5 seconds for action id : {action_id}')
    # await asyncio.sleep(5)
    # return [0,1,2,3,4,5,6,7]

    # loop till we get response of our sample
    while True:
      try:
        response = service.call(
            lambda s, meta: s.GetOutcome(request, 300, metadata=meta)
        )

        # when worker finished fvs run of that sample
        if(response.status == service_pb2.GetOutcomeResponse.Status.COMPLETED):
          # outcome_list = []
          for outcome in response.outcome:
            outcome_dict = {}
            outcome_dict['reward'] = outcome.reward
            outcome_dict['action'] = param_proto_to_dict(outcome.action_attrs)
            outcome_dict['outcome'] = param_proto_to_dict(outcome.outcome_attrs)
            outcome_dict['attributes'] = param_proto_to_dict(outcome.attributes)
            # outcome_list.append(outcome_dict)
          print('outcome_dict : ', outcome_dict)
          return outcome_dict
          # exit the loop
          # break
        # when our sample is in pending or active state at server, try again
        else:
          print(response.response_str)
          await asyncio.sleep(30)
      except Exception as e:
        raise e


async def propose_actions(sight, action_dict):
    # tasks = []
    # iterate over all the samples
    # print(f"actions_dict : {actions_dict}")
    with Attribute('Managed', '0', sight):
      unique_action_id1 = decision.propose_actions(sight, action_dict)
    with Attribute('Managed', '1', sight):
      unique_action_id2 = decision.propose_actions(sight, action_dict)
        # generate task of get_outcome and move on with next action to propose
    task1 = asyncio.create_task(get_outcome(sight.id, unique_action_id1))
    task2 = asyncio.create_task(get_outcome(sight.id, unique_action_id2))

    # wait till we get outcome of all the samples
    time_series = await asyncio.gather(task1,task2)
    print("time_series :", time_series)
    return time_series
    # calculate diff series
    # appy watermark algorithm
    # return the final series

async def main(argv: Sequence[str]) -> None:
  if len(argv) > 1:
      raise app.UsageError("Too many command-line arguments.")

  sample_list = [{'a1': 1, 'a2': 1}]  #, {'a1': 2, 'a2': 2}
  with get_sight_instance() as sight:
      launch_dummy_optimizer(sight)

      # spawn workers
      trials.start_jobs(
              num_train_workers=1,
              num_trials=2,
              binary_path='fvs_sight/fvs_worker.py',
              optimizer_type='worklist_scheduler',
              docker_image='gcr.io/cameltrain/sight-portfolio-worker',
              decision_mode='train',
              deployment_mode='worker_mode',
              worker_mode='dsub_cloud_worker',
              sight=sight,
          )
      print('going to sleep for 5 minutes')
      time.sleep(300)

      with Block("Propose actions", sight):
        with Attribute("project_id", "APR107", sight):
          tasks = []
          for id in range(len(sample_list)):
            with Attribute("sample_id", id, sight):
              tasks.append(asyncio.create_task(propose_actions(sight, sample_list[id])))

          print("waiting for all get outcome to finish.....")
          diff_time_series_all_samples = await asyncio.gather(*tasks)
          print(diff_time_series_all_samples)

def main_wrapper(argv):
    asyncio.run(main(argv))


if __name__ == "__main__":
    app.run(main_wrapper)
