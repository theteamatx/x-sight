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

import warnings


def warn(*args, **kwargs):
    pass


warnings.warn = warn

import os
from typing import Sequence

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

FLAGS = flags.FLAGS


def get_sight_instance():
    params = sight_pb2.Params(
        label="kokua_experiment",
        bucket_name=f'{os.environ["PROJECT_ID"]}-sight',
    )
    sight_obj = Sight(params)
    return sight_obj


action_attrs = {
    "a":
    sight_pb2.DecisionConfigurationStart.AttrProps(
        min_value=0,
        max_value=1,
    ),
    "b":
    sight_pb2.DecisionConfigurationStart.AttrProps(
        min_value=0,
        max_value=1,
    ),
}
# outcome_attrs = {
#     "c":
#     sight_pb2.DecisionConfigurationStart.AttrProps(
#         min_value=0,
#         max_value=1,
#     ),
# }


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
    # decision._attr_dict_to_proto(outcome_attrs,
    #                              decision_configuration.outcome_attrs)
    trials.launch(
        optimizer_object,
        decision_configuration,
        FLAGS.num_train_workers,
        sight,
    )


def simulate_fvs(sight,params_dict):
    print('here params_dict is :', params_dict)
    mitigation_list = [101, 102, 103, 104, 105]
    sim_stream = pd.Series(mitigation_list)
    # print(sim_stream)
    return sim_stream

def driver_func(sight):

  params_dict = decision.decision_point("label", sight)
  # params_dict = {'fvs_type':'managed','region':'BM','project_id':'ACR173','desc': 'fire_projectACR173', 'fire-SIMFIRE_27-1_cycle': 2028, 'fire-SIMFIRE_27-6_stand_area_burned': 10.0, 'fire-SIMFIRE_30-1_cycle': 2031, 'fire-SIMFIRE_30-6_stand_area_burned': 10.0, 'fire-SIMFIRE_31-1_cycle': 2032, 'fire-SIMFIRE_31-6_stand_area_burned': 10.0}
  print('params_dict : ', params_dict)
  # raise SystemError

  sim_stream = simulate_fvs(sight,params_dict)

  outcome = {'time_series' : sim_stream}
  print("outcome : ", outcome)

  decision.decision_outcome('outcome_label', sight, reward=0, outcome=outcome)

def main(argv: Sequence[str]) -> None:
    if len(argv) > 1:
        raise app.UsageError("Too many command-line arguments.")

    with get_sight_instance() as sight:

      launch_dummy_optimizer(sight)

      with Block("Propose actions", sight):
        with Attribute("project_id", "APR107", sight):
          with Attribute("sample_id", "Id-1", sight):
            with Attribute("managed", "1", sight):
              # get actions containing fvs params from the fire model
              actions_dict = {'a': 1,'b': 2}
              unique_action_id = decision.propose_actions(sight, actions_dict)
              print("unique_action_id : ", unique_action_id)
            with Attribute("managed", "0", sight):
              # get actions containing fvs params from the fire model
              actions_dict = {'a': 1,'b': 2}
              unique_action_id = decision.propose_actions(sight, actions_dict)
              print("unique_action_id : ", unique_action_id)

      # # spawn workers
      # trials.start_jobs(
      #         num_train_workers=1,
      #         num_trials=1,
      #         binary_path='kokua-worker.py',
      #         optimizer_type='worklist_scheduler',
      #         docker_image='gcr.io/cameltrain/sight-portfolio-worker',
      #         decision_mode='train',
      #         deployment_mode='worker_mode',
      #         worker_mode='dsub_cloud_worker',
      #         sight=sight,
      #     )


if __name__ == "__main__":
    app.run(main)
