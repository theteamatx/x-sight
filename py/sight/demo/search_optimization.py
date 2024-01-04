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

"""Demo of using the Sight Decision API to optimize an application."""

import math
import random
from typing import Sequence

from absl import app
from absl import logging
from sight import data_structures
from sight.proto import sight_pb2
from sight.sight import Sight
from sight.widgets.decision import decision


def diff_abs(x) -> float:
  """Differentiable variant of the absolute value function."""
  return math.sqrt(x * x + 0.1)


def driver(sight: Sight) -> None:
  """Executes the logic of searching for a value.

  Args:
    sight: The Sight logger object used to drive decisions.
  """
  for i in range(1):
    target = random.randrange(0, 1000)
    current = random.randrange(0, 1000)

    step = 0
    data_structures.log_var('current', current, sight)
    data_structures.log_var('target', target, sight)

    step += 1
    while current != target and step < 100:
      decision.decision_outcome('distance', -diff_abs(target - current), sight)

      data_structures.log_var('current', current, sight)
      choice = decision.decision_point(
          'move',
          sight,
          lambda: {
              'go1': (
                  random.randrange(
                      current // 2 if current < target else target // 2,
                      current * 2 if current > target else target * 2,
                  )
                  - current
              ),
              # 'go2':
              #     random.randrange(
              #         current // 2
              #         if current < target else target // 2, current * 2
              #         if current > target else target * 2) - current,
              # f'{math.ceil((target - current)/2) if target > current else math.floor((target - current)/2)}'
          },
      )
      logging.info('choice=%s', choice)

      current += int(
          choice['go1']
      )  # + choice['go2'])  #int((choice*2 - 1)*100)
      logging.info(
          '%d: %d: amount=%s, current=%s, target=%s',
          i,
          step,
          int(choice['go1']),
          #  int(choice['go2']),
          current,
          target,
      )
      step += 1
    decision.decision_outcome('distance', -diff_abs(target - current), sight)


def main(argv: Sequence[str]) -> None:
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  params = sight_pb2.Params(
      label='SearchOptimization',
      log_owner='bronevet@google.com',
      # local=True,
      capacitor_output=True,
      log_dir_path='/tmp/',
  )

  with Sight(params) as sight:
    decision.run(
        driver_fn=driver,
        state_attrs={
            'current': sight_pb2.DecisionConfigurationStart.AttrProps(
                min_value=0, max_value=1000
            ),
            'target': sight_pb2.DecisionConfigurationStart.AttrProps(
                min_value=0, max_value=1000
            ),
        },
        action_attrs={
            'go1': sight_pb2.DecisionConfigurationStart.AttrProps(
                min_value=0, max_value=100
            ),
        },
        sight=sight,
    )


if __name__ == '__main__':
  app.run(main)
