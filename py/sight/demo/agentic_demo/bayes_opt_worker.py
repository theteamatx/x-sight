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
"""Demo of using the Baysian Optimizer locally with tooling support of Langchain."""

import os
from typing import Sequence, Any
from absl.flags import _exceptions
from bayes_opt import BayesianOptimization
from bayes_opt import UtilityFunction
from langchain_core.tools import Tool
from absl import app
from absl import flags
from sight.proto import sight_pb2
from sight.sight import Sight
from helpers.logs.logs_handler import logger as logging

FLAGS = flags.FLAGS

def get_question_label():
  return 'Q_label1'


# Define the black box function to optimize.
def black_box_function(args):
  return sum(xi**2 for xi in args)


def driver(sight: Sight, bayes_obj: Any) -> Any:
  """Executes the logic of searching for a value.

  Args:
    sight: The Sight logger object used to drive decisions.
  """

  for _ in range(1):
    next_point = bayes_obj.suggest(
        UtilityFunction(kind='ucb', kappa=1.96, xi=0.01))
    logging.info('next_point : %s', next_point)

    reward = black_box_function(list(next_point.values()))
    print('reward : ', reward)

    bayes_obj.register(params=next_point, target=reward)

  return next_point, reward


def get_sight_instance():
  params = sight_pb2.Params(
      label=get_question_label(),
      bucket_name=f'{os.environ["PROJECT_ID"]}-sight',
  )
  sight_obj = Sight(params)
  return sight_obj


# Creating Bayesian Optimizer Object locally
def create_bayes_opt_obj(actions):
  bayes_obj = BayesianOptimization(
      f=None,
      pbounds={
          key: (p["min_value"], p["max_value"]) for key, p in actions.items()
      },
      verbose=2,
      allow_duplicate_points=True,
      # random_state=1,
  )
  return bayes_obj


# @tool(args_schema=None)
def run_BO_experiment(_: str = ""):
  """
  Runs a Bayesian Optimization experiment on the Sphere function.

  This function optimizes over two action attributes ('a1' and 'a2') within the range [2, 5]
  to find the input values that maximize the Sphere function.

  Returns:
      A tuple:
          - actions (dict): The values of 'a1' and 'a2' that produced the highest output.
          - reward (float): The maximum value obtained from the Sphere function using those actions.
  """

  # Initialize absl FLAGS manually if needed
  try:
      flags.FLAGS.mark_as_parsed()
  except _exceptions.DuplicateFlagError:
      pass  # Already parsed
  except _exceptions.UnparsedFlagAccessError:
      flags.FLAGS(['run_BO_experiment'])

  with get_sight_instance() as sight:

    # configuration for the action attributes
    actions = {
        'a1': {
            'min_value': 2,
            'max_value': 5
        },
        'a2': {
            'min_value': 2,
            'max_value': 5
        }
    }
    bo_obj = create_bayes_opt_obj(actions)

    max_reward = 0
    best_actions = {}

    # Iterating over the driver function for 10 iteration to get the optimal result
    for _ in range(0, 10):
      actions_taken, reward = driver(sight, bo_obj)
      if (reward > max_reward):
        max_reward = reward
        best_actions = actions_taken

    # return f"max reward obtained is {reward} with the following actions {str(best_actions)}"
    return f"Max reward obtained is {max_reward} at input actions {best_actions}"

# Tooling the BO experiment
run_BO_tool = Tool(
    name="run_BO_experiment",
    func=run_BO_experiment,
    description=
    "Runs Bayesian optimization on the sphere function and returns the maximum reward and corresponding actions."
)


def main(argv: Sequence[str]) -> None:
  if len(argv) > 1:
    raise app.UsageError("Too many command-line arguments.")

  response = run_BO_experiment()
  print(response)


if __name__ == "__main__":
  app.run(main)
