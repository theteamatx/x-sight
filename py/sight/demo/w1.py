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

FLAGS = flags.FLAGS

# sample = {
#     'project_id': '133a6365-01cf-4b5e-8197-d4779e5ce25c',
#     'fire-SIMFIRE_0-6_stand_area_burned': 100,
#     'fire-SIMFIRE_0-1_cycle': 2013,
#     'fire-SIMFIRE_1-6_stand_area_burned': 71,
#     'fire-SIMFIRE_1-1_cycle': 2014,
#     'fire-SIMFIRE_2-6_stand_area_burned': 100,
#     'fire-SIMFIRE_2-1_cycle': 2015,
#     'fire-SIMFIRE_4-6_stand_area_burned': 10,
#     'fire-SIMFIRE_4-1_cycle': 2017,
#     'fire-SIMFIRE_11-6_stand_area_burned': 80,
#     'fire-SIMFIRE_11-1_cycle': 2024,
#     'fire-SIMFIRE_17-6_stand_area_burned': 45,
#     'fire-SIMFIRE_17-1_cycle': 2030,
#     'fire-SIMFIRE_19-6_stand_area_burned': 45,
#     'fire-SIMFIRE_19-1_cycle': 2032,
#     'fire-SIMFIRE_20-6_stand_area_burned': 21,
#     'fire-SIMFIRE_20-1_cycle': 2033,
#     'fire-SIMFIRE_22-6_stand_area_burned': 34,
#     'fire-SIMFIRE_22-1_cycle': 2035,
#     'fire-SIMFIRE_23-6_stand_area_burned': 100,
#     'fire-SIMFIRE_23-1_cycle': 2036,
#     'fire-SIMFIRE_26-6_stand_area_burned': 16,
#     'fire-SIMFIRE_26-1_cycle': 2039,
#     'fire-SIMFIRE_28-6_stand_area_burned': 100,
#     'fire-SIMFIRE_28-1_cycle': 2041,
#     'fire-SIMFIRE_29-6_stand_area_burned': 7,
#     'fire-SIMFIRE_29-1_cycle': 2042,
#     'fire-SIMFIRE_33-6_stand_area_burned': 100,
#     'fire-SIMFIRE_33-1_cycle': 2046,
#     'fire-SIMFIRE_35-6_stand_area_burned': 87,
#     'fire-SIMFIRE_35-1_cycle': 2048,
#     'fire-SIMFIRE_36-6_stand_area_burned': 53,
#     'fire-SIMFIRE_36-1_cycle': 2049,
#     'fire-SIMFIRE_37-6_stand_area_burned': 51,
#     'fire-SIMFIRE_37-1_cycle': 2050,
#     'fire-SIMFIRE_39-6_stand_area_burned': 8,
#     'fire-SIMFIRE_39-1_cycle': 2052,
#     'fire-SIMFIRE_42-6_stand_area_burned': 100,
#     'fire-SIMFIRE_42-1_cycle': 2055,
#     'fire-SIMFIRE_43-6_stand_area_burned': 95,
#     'fire-SIMFIRE_43-1_cycle': 2056,
#     'fire-SIMFIRE_44-6_stand_area_burned': 14,
#     'fire-SIMFIRE_44-1_cycle': 2057,
#     'fire-SIMFIRE_45-6_stand_area_burned': 18,
#     'fire-SIMFIRE_45-1_cycle': 2058,
#     'fire-SIMFIRE_47-6_stand_area_burned': 100,
#     'fire-SIMFIRE_47-1_cycle': 2060,
#     'fire-SIMFIRE_49-6_stand_area_burned': 100,
#     'fire-SIMFIRE_49-1_cycle': 2062,
#     'fire-SIMFIRE_50-6_stand_area_burned': 25,
#     'fire-SIMFIRE_50-1_cycle': 2063,
#     'fire-SIMFIRE_53-6_stand_area_burned': 100,
#     'fire-SIMFIRE_53-1_cycle': 2066,
#     'fire-SIMFIRE_54-6_stand_area_burned': 66,
#     'fire-SIMFIRE_54-1_cycle': 2067,
#     'fire-SIMFIRE_56-6_stand_area_burned': 63,
#     'fire-SIMFIRE_56-1_cycle': 2069,
#     'fire-SIMFIRE_59-6_stand_area_burned': 45,
#     'fire-SIMFIRE_59-1_cycle': 2072,
#     'fire-SIMFIRE_60-6_stand_area_burned': 79,
#     'fire-SIMFIRE_60-1_cycle': 2073,
#     'fire-SIMFIRE_63-6_stand_area_burned': 80,
#     'fire-SIMFIRE_63-1_cycle': 2076,
#     'fire-SIMFIRE_64-6_stand_area_burned': 47,
#     'fire-SIMFIRE_64-1_cycle': 2077,
#     'fire-SIMFIRE_65-6_stand_area_burned': 64,
#     'fire-SIMFIRE_65-1_cycle': 2078,
#     'fire-SIMFIRE_66-6_stand_area_burned': 100,
#     'fire-SIMFIRE_66-1_cycle': 2079,
#     'fire-SIMFIRE_68-6_stand_area_burned': 100,
#     'fire-SIMFIRE_68-1_cycle': 2081,
#     'fire-SIMFIRE_70-6_stand_area_burned': 30,
#     'fire-SIMFIRE_70-1_cycle': 2083,
#     'fire-SIMFIRE_71-6_stand_area_burned': 12,
#     'fire-SIMFIRE_71-1_cycle': 2084,
#     'fire-SIMFIRE_72-6_stand_area_burned': 51,
#     'fire-SIMFIRE_72-1_cycle': 2085,
#     'fire-SIMFIRE_75-6_stand_area_burned': 17,
#     'fire-SIMFIRE_75-1_cycle': 2088,
#     'fire-SIMFIRE_76-6_stand_area_burned': 100,
#     'fire-SIMFIRE_76-1_cycle': 2089,
#     'fire-SIMFIRE_79-6_stand_area_burned': 60,
#     'fire-SIMFIRE_79-1_cycle': 2092,
#     'fire-SIMFIRE_81-6_stand_area_burned': 45,
#     'fire-SIMFIRE_81-1_cycle': 2094,
#     'fire-SIMFIRE_84-6_stand_area_burned': 100,
#     'fire-SIMFIRE_84-1_cycle': 2097,
#     'fire-SIMFIRE_88-6_stand_area_burned': 58,
#     'fire-SIMFIRE_88-1_cycle': 2101,
#     'fire-SIMFIRE_90-6_stand_area_burned': 82,
#     'fire-SIMFIRE_90-1_cycle': 2103,
#     'fire-SIMFIRE_92-6_stand_area_burned': 60,
#     'fire-SIMFIRE_92-1_cycle': 2105,
#     'fire-SIMFIRE_94-6_stand_area_burned': 56,
#     'fire-SIMFIRE_94-1_cycle': 2107,
#     'fire-SIMFIRE_96-6_stand_area_burned': 100,
#     'fire-SIMFIRE_96-1_cycle': 2109,
#     'fire-SIMFIRE_97-6_stand_area_burned': 3,
#     'fire-SIMFIRE_97-1_cycle': 2110,
#     'fire-SIMFIRE_98-6_stand_area_burned': 87,
#     'fire-SIMFIRE_98-1_cycle': 2111,
#     'region': 'NC',
#     'base-FERTILIZ-howManyCycle': 1.0,
#     'base-FERTILIZ-extra_step': 0.0,
#     'base-FERTILIZ-extra_offset': 0.0
# }

sample = {'base-FERTILIZ-extra_offset': 0.0}


def get_question_label_to_propose_actions():
  return 'Q_label3'


def get_question_label():
  return 'Q_label1'


# Define the black box function to optimize.
def black_box_function(args):
  return sum(xi**2 for xi in args)


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


async def propose_actions_wrapper(sight: Sight, question_label: str) -> None:
  # sample_list = [sample for i in range(num_trials)]

  with Block("Propose actions", sight):
    with Attribute("project_id", "APR107", sight):
      tasks = []
      # print("len(sample_list) : ", len(sample_list))
      # for id in range(len(sample_list)):
      with Attribute("sample_id", 'sample_1', sight):
        tasks.append(
            sight.create_task(
                # both base and treatment are considerred to be same dict here
                propose_actions(sight, question_label, sample, sample)))

      print("waiting for all get outcome to finish.....")
      diff_time_series = await asyncio.gather(*tasks)
      print("all get outcome are finished.....")
      print(f'Combine Series : {diff_time_series}')


def driver(sight: Sight) -> None:
  """Executes the logic of searching for a value.

  Args:
    sight: The Sight logger object used to drive decisions.
  """

  for _ in range(1):
    next_point = decision.decision_point(get_question_label(), sight)
    print('next_point : ', next_point)

    # using next_points to propose actions
    asyncio.run(
        propose_actions_wrapper(sight, get_question_label_to_propose_actions()))

    reward = black_box_function(list(next_point.values()))
    print('reward : ', reward)
    decision.decision_outcome(json.dumps(next_point), sight, reward)


def get_sight_instance():
  params = sight_pb2.Params(
      label=get_question_label(),
      bucket_name=f'{os.environ["PROJECT_ID"]}-sight',
  )
  sight_obj = Sight(params)
  return sight_obj


def main(argv: Sequence[str]) -> None:
  if len(argv) > 1:
    raise app.UsageError("Too many command-line arguments.")

  # num_attributes = 10
  # attr_range = (0, 5)
  # action_attrs = {}
  # for i in range(num_attributes):
  #   key = f"{i}"  # Generate unique keys
  #   action_attrs[key] = sight_pb2.DecisionConfigurationStart.AttrProps(
  #       min_value=attr_range[0],
  #       max_value=attr_range[1],
  #   )

  with get_sight_instance() as sight:

    if (FLAGS.parent_id):
      sight_obj = sight_pb2.Object()
      sight_obj.sub_type = sight_pb2.Object.SubType.ST_LINK
      sight_obj.link.linked_sight_id = FLAGS.parent_id
      sight_obj.link.link_type = sight_pb2.Link.LinkType.LT_CHILD_TO_PARENT
      frame = inspect.currentframe().f_back.f_back.f_back
      sight.set_object_code_loc(sight_obj, frame)
      sight.log_object(sight_obj, True)

    decision.run(sight=sight,
                 question_label=get_question_label(),
                 driver_fn=driver)


if __name__ == "__main__":
  app.run(main)
