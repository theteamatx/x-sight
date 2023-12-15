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

import os
import random
from typing import Sequence

from absl import app
from absl import flags
from absl import logging
from sight.proto import sight_pb2
from sight.sight import Sight
from sight.widgets.decision import decision

FLAGS = flags.FLAGS

def driver(sight: Sight) -> None:
  """Executes the logic of searching for a value.

  Args:
    sight: The Sight logger object used to drive decisions.
  """
  secret_num = random.randrange(0, 1000)
  logging.info('secret_num=%s', secret_num)
  choice = decision.decision_point('move', sight)
  logging.info('choice=%s, error=%s', choice, choice['guess'] - secret_num)

  decision.decision_outcome(
      'distance', -abs(choice['guess'] - secret_num), sight
  )

  proposed_guess = secret_num + (choice['guess'] - secret_num) / 2
  logging.info('proposed_guess=%s', proposed_guess)
  decision.propose_action(
      -abs(choice['guess'] - secret_num) / 2, {'guess': proposed_guess}, sight
  )

def get_sight_instance():
  params = sight_pb2.Params(
      label='secret_find_experiment',
      bucket_name=f'{os.environ["PROJECT_ID"]}-sight',
  )
  sight_obj = Sight(params)
  return sight_obj

def main(argv: Sequence[str]) -> None:
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  with get_sight_instance() as sight:
    decision.run(
        driver_fn=driver,
        state_attrs={},
        action_attrs={
            'guess': sight_pb2.DecisionConfigurationStart.AttrProps(
                min_value=0, max_value=1000, step_size=10
            ),
        },
        sight=sight,
    )


if __name__ == '__main__':
  app.run(main)
