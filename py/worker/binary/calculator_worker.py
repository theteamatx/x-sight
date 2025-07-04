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
"""Worker script to be run via calculator problem."""
from typing import Any, Dict, Tuple

from absl import app
from sight import sight


# Question mapped to calculator problem
def get_question_label():
  return "Calculator"


def main(action: Dict[str, Any]) -> Tuple[float, Dict[str, Any]]:
  ops = action["operator"]
  v1 = action["operand1"]
  v2 = action["operand2"]
  if ops.lower() == "add":
    result = v1 + v2
  elif ops.lower() == "subtract":
    result = v1 - v2
  elif ops.lower() == "multiply":
    result = v1 * v2
  elif ops.lower() == "divide":
    result = v1 / v2
  else:
    result = "not supported operation by this calculator"

  outcome = {"final_result": result}
  return 1, outcome


if __name__ == "__main__":
  app.run(lambda _: sight.run_worker(
      main,
      {
          "label": get_question_label(),
      },
  ))
