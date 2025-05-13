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

import json
import os
import random
import time
from typing import Sequence

from absl import app
from absl import flags
from sight.proto import sight_pb2
from sight import sight
from sight.sight import Sight
from sight.widgets.decision import decision


# Question mapped to calculator problem
def get_question_label():
  return 'calculator'


# Primary logic to be handeled via calculator worker
def calculator(v1, v2, ops):
  if (ops.lower() == "add"):
    return v1 + v2
  elif (ops.lower() == "subtract"):
    return v1 - v2
  elif (ops.lower() == "multiply"):
    return v1 * v2
  elif (ops.lower() == "divide"):
    return v1 / v2
  else:
    return "not supported operation by this calculator"


def driver_fn(sight):

  params_dict = decision.decision_point(get_question_label(), sight)
  print("params_dict here is : ", params_dict)

  result = calculator(params_dict["v1"], params_dict["v2"], params_dict["ops"])

  outcome = {'result': result}
  print("outcome : ", outcome)
  decision.decision_outcome('outcome_label', sight, reward=0, outcome=outcome)


def main(argv: Sequence[str]) -> None:
  if len(argv) > 1:
    raise app.UsageError("Too many command-line arguments.")

  # Enry point for the worker to start asking for the calculator related actions
  sight.run_worker(question_label=get_question_label(), driver_fn=driver_fn)


if __name__ == "__main__":
  app.run(main)
