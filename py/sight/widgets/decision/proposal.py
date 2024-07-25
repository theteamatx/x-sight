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


import asyncio
from sight.attribute import Attribute
from sight.block import Block
from absl import flags
from sight.proto import sight_pb2
from sight.sight import Sight
from sight.widgets.decision import decision
from sight.widgets.decision import trials
from fvs_sight.fvs_api import action_attrs
from fvs_sight.fvs_api import outcome_attrs
from sight.widgets.decision.single_action_optimizer_client import (
    SingleActionOptimizerClient)
from sight.widgets.decision.resource_lock import RWLockDictWrapper

global_outcome_mapping = RWLockDictWrapper()

FLAGS = flags.FLAGS


"""helper functions to be used in portfolio flow ."""

def launch_worklist_scheduler(sight):
    optimizer_object = SingleActionOptimizerClient(
        sight_pb2.DecisionConfigurationStart.OptimizerType.
        OT_WORKLIST_SCHEDULER, sight)
    decision_configuration = sight_pb2.DecisionConfigurationStart()
    decision_configuration.optimizer_type = optimizer_object.optimizer_type()

    decision_configuration.num_trials = FLAGS.num_trials
    decision_configuration.choice_config[sight.params.label].CopyFrom(
        optimizer_object.create_config())
    # decision._attr_dict_to_proto(state_attrs,
    #                              decision_configuration.state_attrs)
    decision.attr_dict_to_proto(action_attrs,
                                decision_configuration.action_attrs)
    decision.attr_dict_to_proto(outcome_attrs,
                                decision_configuration.outcome_attrs)
    trials.launch(
        optimizer_object,
        decision_configuration,
        FLAGS.num_train_workers,
        sight,
    )

def spawn_workers(sight):
    trials.start_jobs(
            num_train_workers=1,
            num_trials=4,
            binary_path='fvs_sight/fvs_worker.py',
            optimizer_type='worklist_scheduler',
            docker_image='gcr.io/cameltrain/kokua_portfolio:fvs',
            decision_mode='train',
            deployment_mode='worker_mode',
            worker_mode='dsub_cloud_worker',
            sight=sight,
        )

async def push_message(sight_id, action_id):
    try:
        global_outcome_mapping.set_for_key(action_id, None)
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
