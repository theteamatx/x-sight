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
from typing import Any, Sequence

from absl import app
from absl import flags
# from fvs_sight.fvs_api import action_attrs
# from fvs_sight.fvs_api import outcome_attrs
from fvs_sight import fvs_api
import pandas as pd
from sight.attribute import Attribute
from sight.block import Block
from sight.proto import sight_pb2
from sight.sight import Sight
# from sight.widgets.decision.proposal import spawn_workers, launch_worklist_scheduler, propose_actions
from sight.widgets.decision import decision
from sight.widgets.decision import proposal
from sight.widgets.decision.resource_lock import RWLockDictWrapper

global_outcome_mapping = RWLockDictWrapper()

sample = {
    'region': 'NC',
    # 'project_id': '133a6365-01cf-4b5e-8197-d4779e5ce25c',
    # 'fire-SIMFIRE_0-6_stand_area_burned': 100,
    # 'fire-SIMFIRE_0-1_cycle': 2013,
    # 'fire-SIMFIRE_1-6_stand_area_burned': 71,
    # 'fire-SIMFIRE_1-1_cycle': 2014,
    # 'fire-SIMFIRE_2-6_stand_area_burned': 100,
    # 'fire-SIMFIRE_2-1_cycle': 2015,
    # 'fire-SIMFIRE_4-6_stand_area_burned': 10,
    # 'fire-SIMFIRE_4-1_cycle': 2017,
    # 'fire-SIMFIRE_11-6_stand_area_burned': 80,
    # 'fire-SIMFIRE_11-1_cycle': 2024,
    # 'fire-SIMFIRE_17-6_stand_area_burned': 45,
    # 'fire-SIMFIRE_17-1_cycle': 2030,
    # 'fire-SIMFIRE_19-6_stand_area_burned': 45,
    # 'fire-SIMFIRE_19-1_cycle': 2032,
    # 'fire-SIMFIRE_20-6_stand_area_burned': 21,
    # 'fire-SIMFIRE_20-1_cycle': 2033,
    # 'fire-SIMFIRE_22-6_stand_area_burned': 34,
    # 'fire-SIMFIRE_22-1_cycle': 2035,
    # 'fire-SIMFIRE_23-6_stand_area_burned': 100,
    # 'fire-SIMFIRE_23-1_cycle': 2036,
    # 'fire-SIMFIRE_26-6_stand_area_burned': 16,
    # 'fire-SIMFIRE_26-1_cycle': 2039,
    # 'fire-SIMFIRE_28-6_stand_area_burned': 100,
    # 'fire-SIMFIRE_28-1_cycle': 2041,
    # 'fire-SIMFIRE_29-6_stand_area_burned': 7,
    # 'fire-SIMFIRE_29-1_cycle': 2042,
    # 'fire-SIMFIRE_33-6_stand_area_burned': 100,
    # 'fire-SIMFIRE_33-1_cycle': 2046,
    # 'fire-SIMFIRE_35-6_stand_area_burned': 87,
    # 'fire-SIMFIRE_35-1_cycle': 2048,
    # 'fire-SIMFIRE_36-6_stand_area_burned': 53,
    # 'fire-SIMFIRE_36-1_cycle': 2049,
    # 'fire-SIMFIRE_37-6_stand_area_burned': 51,
    # 'fire-SIMFIRE_37-1_cycle': 2050,
    # 'fire-SIMFIRE_39-6_stand_area_burned': 8,
    # 'fire-SIMFIRE_39-1_cycle': 2052,
    # 'fire-SIMFIRE_42-6_stand_area_burned': 100,
    # 'fire-SIMFIRE_42-1_cycle': 2055,
    # 'fire-SIMFIRE_43-6_stand_area_burned': 95,
    # 'fire-SIMFIRE_43-1_cycle': 2056,
    # 'fire-SIMFIRE_44-6_stand_area_burned': 14,
    # 'fire-SIMFIRE_44-1_cycle': 2057,
    # 'fire-SIMFIRE_45-6_stand_area_burned': 18,
    # 'fire-SIMFIRE_45-1_cycle': 2058,
    # 'fire-SIMFIRE_47-6_stand_area_burned': 100,
    # 'fire-SIMFIRE_47-1_cycle': 2060,
    # 'fire-SIMFIRE_49-6_stand_area_burned': 100,
    # 'fire-SIMFIRE_49-1_cycle': 2062,
    # 'fire-SIMFIRE_50-6_stand_area_burned': 25,
    # 'fire-SIMFIRE_50-1_cycle': 2063,
    # 'fire-SIMFIRE_53-6_stand_area_burned': 100,
    # 'fire-SIMFIRE_53-1_cycle': 2066,
    # 'fire-SIMFIRE_54-6_stand_area_burned': 66,
    # 'fire-SIMFIRE_54-1_cycle': 2067,
    # 'fire-SIMFIRE_56-6_stand_area_burned': 63,
    # 'fire-SIMFIRE_56-1_cycle': 2069,
    # 'fire-SIMFIRE_59-6_stand_area_burned': 45,
    # 'fire-SIMFIRE_59-1_cycle': 2072,
    # 'fire-SIMFIRE_60-6_stand_area_burned': 79,
    # 'fire-SIMFIRE_60-1_cycle': 2073,
    # 'fire-SIMFIRE_63-6_stand_area_burned': 80,
    # 'fire-SIMFIRE_63-1_cycle': 2076,
    # 'fire-SIMFIRE_64-6_stand_area_burned': 47,
    # 'fire-SIMFIRE_64-1_cycle': 2077,
    # 'fire-SIMFIRE_65-6_stand_area_burned': 64,
    # 'fire-SIMFIRE_65-1_cycle': 2078,
    # 'fire-SIMFIRE_66-6_stand_area_burned': 100,
    # 'fire-SIMFIRE_66-1_cycle': 2079,
    # 'fire-SIMFIRE_68-6_stand_area_burned': 100,
    # 'fire-SIMFIRE_68-1_cycle': 2081,
    # 'fire-SIMFIRE_70-6_stand_area_burned': 30,
    # 'fire-SIMFIRE_70-1_cycle': 2083,
    # 'fire-SIMFIRE_71-6_stand_area_burned': 12,
    # 'fire-SIMFIRE_71-1_cycle': 2084,
    # 'fire-SIMFIRE_72-6_stand_area_burned': 51,
    # 'fire-SIMFIRE_72-1_cycle': 2085,
    # 'fire-SIMFIRE_75-6_stand_area_burned': 17,
    # 'fire-SIMFIRE_75-1_cycle': 2088,
    # 'fire-SIMFIRE_76-6_stand_area_burned': 100,
    # 'fire-SIMFIRE_76-1_cycle': 2089,
    # 'fire-SIMFIRE_79-6_stand_area_burned': 60,
    # 'fire-SIMFIRE_79-1_cycle': 2092,
    # 'fire-SIMFIRE_81-6_stand_area_burned': 45,
    # 'fire-SIMFIRE_81-1_cycle': 2094,
    # 'fire-SIMFIRE_84-6_stand_area_burned': 100,
    # 'fire-SIMFIRE_84-1_cycle': 2097,
    # 'fire-SIMFIRE_88-6_stand_area_burned': 58,
    # 'fire-SIMFIRE_88-1_cycle': 2101,
    # 'fire-SIMFIRE_90-6_stand_area_burned': 82,
    # 'fire-SIMFIRE_90-1_cycle': 2103,
    # 'fire-SIMFIRE_92-6_stand_area_burned': 60,
    # 'fire-SIMFIRE_92-1_cycle': 2105,
    # 'fire-SIMFIRE_94-6_stand_area_burned': 56,
    # 'fire-SIMFIRE_94-1_cycle': 2107,
    # 'fire-SIMFIRE_96-6_stand_area_burned': 100,
    # 'fire-SIMFIRE_96-1_cycle': 2109,
    # 'fire-SIMFIRE_97-6_stand_area_burned': 3,
    # 'fire-SIMFIRE_97-1_cycle': 2110,
    # 'fire-SIMFIRE_98-6_stand_area_burned': 87,
    # 'fire-SIMFIRE_98-1_cycle': 2111,
    # 'base-FERTILIZ-howManyCycle': 1.0,
    # 'base-FERTILIZ-extra_step': 0.0,
    # 'base-FERTILIZ-extra_offset': 0.0
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

  x_start_time = time.perf_counter()
  print(f"Proposing Start m-um ")

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
  print(f"Propose actions m-um took {x_end_time - x_start_time:.4f} seconds.")
  return unmanaged_response, managed_response


async def main(sight: Sight, argv: Sequence[str]) -> None:
  if len(argv) > 1:
    raise app.UsageError("Too many command-line arguments.")

  sample_list = [sample for i in range(FLAGS.num_trials)]

  # print('SIGHT ID => ',sight.id)
  with Block("Propose actions", sight):
    with Attribute("project_id", "APR107", sight):
      tasks = []
      print("len(sample_list) : ", len(sample_list))

      x_start_time = time.perf_counter()
      print(f"Proposing Start ")

      for id in range(len(sample_list)):
        with Attribute("sample_id", id, sight):
          tasks.append(
              sight.create_task(
                  # both base and treatment are considerred to be same dict here
                  propose_actions(sight, sample_list[id], sample_list[id])))

      x_end_time = time.perf_counter()
      print(f"Propose actions took {x_end_time - x_start_time:.4f} seconds.")

      diff_time_series = await asyncio.gather(*tasks)

      print("waiting for all get outcome to finish.....")
      print("all get outcome are finished.....")
      print(f'Combine Series : {diff_time_series}')


def main_wrapper(argv):
  with get_sight_instance() as sight:
    decision.run(action_attrs=fvs_api.get_action_attrs(),
                 outcome_attrs=fvs_api.get_outcome_attrs(),
                 sight=sight)
    start_time = time.perf_counter()
    sleep_time_in_min = 15
    print(f"Waiting for {sleep_time_in_min} min for workers to start ...")
    time.sleep(sleep_time_in_min * 60)
    asyncio.run(main(sight, argv))
    end_time = time.perf_counter()
    elapsed_time = end_time - start_time
    print(f"Elapsed time: {elapsed_time} seconds")
    hours, remainder = divmod(elapsed_time, 3600)
    minutes, seconds = divmod(remainder, 60)

    if hours > 0:
      print(
          f"Elapsed time: {int(hours)} hour(s), {int(minutes)} minute(s), {seconds:.2f} second(s)"
      )
    elif minutes > 0:
      print(f"Elapsed time: {int(minutes)} minute(s), {seconds:.2f} second(s)")
    else:
      print(f"Elapsed time: {seconds:.2f} second(s)")


if __name__ == "__main__":
  app.run(main_wrapper)
