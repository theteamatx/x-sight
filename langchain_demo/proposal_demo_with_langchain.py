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
import os
from pathlib import Path
from absl import app
from absl import flags
import numpy as np
import pandas as pd
from langchain_core.tools import tool
from sight.attribute import Attribute
from sight.block import Block
from absl.flags import _exceptions
from sight import data_structures
from sight.proto import sight_pb2
from sight.sight import Sight
from sight.widgets.decision import decision
from sight.widgets.decision import proposal
from sight.widgets.decision import utils
from helpers.logs.logs_handler import logger as logging

FLAGS = flags.FLAGS


def get_question_label():
  return 'Q_label4'

async def propose_actions(
    sight: Sight,
    question_label: str,
    actions: dict[str, Any],
) -> pd.Series:
  tasks = []
  with Attribute("task", "multiply", sight):
    task = sight.create_task(
        proposal.propose_actions(sight, question_label, action_dict=actions))
    tasks.append(task)

  [final_result] = await asyncio.gather(*tasks)
  return final_result


async def propose_actions_wrapper(sight: Sight, question_label: str, actions: dict) -> str:

  with Block("Propose actions", sight):
    tasks = []
    tasks.append(
        sight.create_task(propose_actions(sight, question_label, actions)))

    logging.info("waiting for all get outcome to finish.....")
    result = await asyncio.gather(*tasks)
    logging.info(f'result : {result}')
    return result


def get_sight_instance():
  params = sight_pb2.Params(
      label=get_question_label(),
      bucket_name=f'{os.environ["PROJECT_ID"]}-sight',
  )
  sight_obj = Sight(params)
  return sight_obj

@tool
def calculator_api_with_sight(a: int, b: int, ops: str) -> str:

  """
  Perform a basic arithmetic operation (addition, subtraction, etc.) on two integers using the Sight backend system.

  This function proposes a calculation action (with inputs `a`, `b`, and the operation `ops`) to the server via a Sight worker.
  It waits for the worker to process the action and return the computed result.

  Args:
      a (int): The first integer operand.
      b (int): The second integer operand.
      ops (str): The operation to perform. Supported operations include:
          - "add" for addition
          - "subtract" for subtraction
          - "multiply" for multiplication
          - "divide" for division

  Returns:
      str: The result of the calculation as a string.

  Example:
      >>> calculator_api_with_sight(5, 3, "add")
      "8"
  """
  # Initialize absl FLAGS manually if needed
  try:
      flags.FLAGS.mark_as_parsed()
      flags.FLAGS.sight_log_id = '2717381483903190742'
      flags.FLAGS.server_mode = 'local'
  except _exceptions.DuplicateFlagError:
      pass  # Already parsed
  except _exceptions.UnparsedFlagAccessError:
      flags.FLAGS(['calculator_api_with_sight'])

  with get_sight_instance() as sight:
    # this thread checks the outcome for proposed action from server
    print("sight.id : ", sight.id)
    # raise SystemError
    decision.init_sight_polling_thread(sight.id, get_question_label())
    actions = {"v1": a, "v2": b, "ops": ops}
    result = asyncio.run(propose_actions_wrapper(sight, get_question_label(), actions))
    return result

if __name__ == "__main__":
  result = calculator_api_with_sight.invoke({"a": 10, "b": 2, "ops": "add"})
  print(result)

