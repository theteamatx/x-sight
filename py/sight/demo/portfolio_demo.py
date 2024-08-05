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
from sight.attribute import Attribute
from sight.block import Block
from sight.proto import sight_pb2
from sight.sight import Sight
from fvs_sight.fvs_api import action_attrs
from fvs_sight.fvs_api import outcome_attrs
from sight.widgets.decision import decision
# from sight.widgets.decision.proposal import spawn_workers, launch_worklist_scheduler, propose_actions
from sight.widgets.decision import proposal
from sight.widgets.decision.resource_lock import RWLockDictWrapper

global_outcome_mapping = RWLockDictWrapper()

def get_sight_instance():
    params = sight_pb2.Params(
        label="kokua_experiment",
        bucket_name=f'{os.environ["PROJECT_ID"]}-sight',
    )
    sight_obj = Sight(params)
    return sight_obj

async def main(sight: Sight, argv: Sequence[str]) -> None:
    if len(argv) > 1:
        raise app.UsageError("Too many command-line arguments.")

    sample_list = [{
        "a1": 1.0,
        "a2": 0.0,
        "a3": "BM"
    }, {
        "a1": 0.5,
        "a2": 0.0,
        "a3": "NC"
    }]

    # print('SIGHT ID => ',sight.id)
    with Block("Propose actions", sight):
        with Attribute("project_id", "APR107", sight):
            tasks = []
            for id in range(len(sample_list)):
                with Attribute("sample_id", id, sight):
                    tasks.append(
                        asyncio.create_task(
                            proposal.propose_actions(sight, sample_list[id])))

            print("waiting for all get outcome to finish.....")
            diff_time_series = await asyncio.gather(*tasks)
            print(f'Combine Series {diff_time_series}')

def main_wrapper(argv):
    with get_sight_instance() as sight:

        decision.run(
            action_attrs=action_attrs,
            outcome_attrs=outcome_attrs,
            sight=sight
        )

        # print('going to sleep for 6 minutes')
        # time.sleep(360)

        # decision.init_sight_polling_thread(sight.id)
        asyncio.run(main(sight, argv))


if __name__ == "__main__":
    app.run(main_wrapper)
