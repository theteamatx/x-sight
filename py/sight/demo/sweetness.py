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

"""Demo of using the Sight Decision API to train sweetness controller."""
import warnings


def warn(*args, **kwargs):
  pass


warnings.warn = warn

import os
import random
from typing import Sequence

from absl import app
from absl import flags
from sight import data_structures
from sight.proto import sight_pb2
from sight.sight import Sight
from sight.widgets.decision import decision

FLAGS = flags.FLAGS

def driver(sight: Sight) -> None:
  """Executes the logic of searching for a value.

  Args:
    sight: The Sight logger object used to drive decisions.
  """
  sweet_tooth = random.randrange(0, 10)
  print("current value of sweet_tooth : ", sweet_tooth)
  data_structures.log_var("sweet_tooth", sweet_tooth, sight)

  for _ in range(1):
    choice = decision.decision_point("candy", sight)
    sight.text(
        "sweet_tooth=%s, choice=%s, joy=%s"
        % (
            sweet_tooth,
            choice["sweetness"],
            float(choice["sweetness"]) * sweet_tooth,
        )
    )

    reward = float(choice["sweetness"]) * sweet_tooth

    decision.decision_outcome("joy", reward, sight)

def get_sight_instance():
  params = sight_pb2.Params(
      label='sweetness_experiment',
      bucket_name=f'{os.environ["PROJECT_ID"]}-sight',
  )
  sight_obj = Sight(params)
  return sight_obj

def main(argv: Sequence[str]) -> None:
  if len(argv) > 1:
    raise app.UsageError("Too many command-line arguments.")

  with get_sight_instance() as sight:
    decision.run(
        driver_fn=driver,
        state_attrs={
            "sweet_tooth": sight_pb2.DecisionConfigurationStart.AttrProps(
                min_value=0,
                max_value=10,
                step_size=1,
            ),
        },
        action_attrs={
            "sweetness": sight_pb2.DecisionConfigurationStart.AttrProps(
                min_value=0,
                max_value=3,
                step_size=1,
            ),
        },
        sight=sight,
    )


if __name__ == "__main__":
  app.run(main)
