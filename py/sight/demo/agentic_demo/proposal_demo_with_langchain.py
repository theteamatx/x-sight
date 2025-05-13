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
from absl import flags
from langchain_core.tools import tool
from absl.flags import _exceptions
from sight.sight import Sight
from sight.widgets.decision import decision
from sight.widgets.decision import proposal

FLAGS = flags.FLAGS


def get_question_label():
  return 'calculator'

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
    flags.FLAGS.server_mode = 'local'
  except _exceptions.DuplicateFlagError:
    pass  # Already parsed
  except _exceptions.UnparsedFlagAccessError:
    flags.FLAGS(['calculator_api_with_sight'])

  # config contains the data from all the config files
  config = decision.DecisionConfig(config_dir_path=FLAGS.config_path)

  with Sight.create('langchain_demo_label', config) as sight:
    actions = {"v1": a, "v2": b, "ops": ops}
    result = asyncio.run(
        proposal.propose_actions(sight, get_question_label(), actions))
    return result


if __name__ == "__main__":
  result = calculator_api_with_sight.invoke({"a": 10, "b": 2, "ops": "add"})
  print(result)
