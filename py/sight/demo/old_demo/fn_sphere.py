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

import inspect
import json
import os
import random
from typing import Sequence

from absl import app
from absl import flags
import numpy as np
from sight import data_structures
from sight.proto import sight_pb2
from sight.sight import Sight
from sight.widgets.decision import decision
from sight.widgets.decision import utils

FLAGS = flags.FLAGS


def get_question_label():
  return 'Q_label1'


# Define the black box function to optimize.
def black_box_function(args):
  return sum(xi**2 for xi in args)


def driver(sight: Sight) -> None:
  """Executes the logic of searching for a value.

  Args:
    sight: The Sight logger object used to drive decisions.
  """

  for _ in range(1):
    next_point = decision.decision_point(get_question_label(), sight)
    print('next_point : ', next_point)
    reward = black_box_function(list(next_point.values()))
    print('reward : ', reward)
    decision.decision_outcome(json.dumps(next_point), sight, reward)


def get_sight_instance():
  params = sight_pb2.Params(
      label='benchmark_experiment',
      bucket_name=f'{os.environ["PROJECT_ID"]}-sight',
  )
  sight_obj = Sight(params)
  return sight_obj


def main(argv: Sequence[str]) -> None:
  if len(argv) > 1:
    raise app.UsageError("Too many command-line arguments.")

  num_attributes = 10
  attr_range = (0, 5)
  action_attrs = {}
  for i in range(num_attributes):
    key = f"{i}"  # Generate unique keys
    action_attrs[key] = sight_pb2.DecisionConfigurationStart.AttrProps(
        min_value=attr_range[0],
        max_value=attr_range[1],
    )

  with get_sight_instance() as sight:

    questions_config = utils.load_yaml_config('fvs_sight/question_config.yaml')
    optimizers_config = utils.load_yaml_config(
        'fvs_sight/optimizer_config.yaml')
    workers_config = utils.load_yaml_config('fvs_sight/worker_config.yaml')

    for question_label, question_config in questions_config.items():
      optimizer_config = optimizers_config[question_label]
      global_config = {
          'question_config': question_config,
          'optimizer_config': optimizer_config,
          'workers_config': workers_config
      }

      decision.run(sight, configs=global_config, question_label=question_label)


if __name__ == "__main__":
  app.run(main)
