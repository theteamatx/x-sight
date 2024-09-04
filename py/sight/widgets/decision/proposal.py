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
from sight.widgets.decision.single_action_optimizer_client import (
    SingleActionOptimizerClient)
from sight.widgets.decision.resource_lock import RWLockDictWrapper

global_outcome_mapping = RWLockDictWrapper()

FLAGS = flags.FLAGS


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
            # print('action_id: ', actions_id, ', outcome=', outcome)
            if outcome is not None:
                # print('action_id: ', actions_id, ', outcome=', outcome, ' RETURNING')
                return outcome
            else:
                # async_dict = global_outcome_mapping.get()
                # print(f'GLOBAL_MAPPING_GET_OUTCOME_QUEUE => {async_dict}')
                time = 5
                print(f'Waiting for {actions_id} for {time} seconds...')
                await asyncio.sleep(time)
        except Exception as e:
            print('action_id: ', actions_id, ', exception=', e)
            raise e
