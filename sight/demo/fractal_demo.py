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

"""Demo of using the fractal environment."""

import os
from typing import Sequence
from absl import app
from absl import flags
from sight import data_structures
from sight.proto import sight_pb2
from sight.sight import Sight
from sight.widgets.decision import decision
import numpy as np
from google3.googlex.fractal.library.rl.fourier_acme.fourier_environment import fourier_environment

FLAGS = flags.FLAGS

def driver_fn(sim_env, sight: Sight) -> None:
  """Driver function to run the loop."""
  initial_state = sim_env.reset()

  state_attrs = decision.get_state_attrs(sight)
  for i in range(len(state_attrs)):
    data_structures.log_var(state_attrs[i], initial_state.observation[i], sight)

  for i in range(3):
    action_dict = decision.decision_point("DP_label", sight)

    action = np.array(list(action_dict.values()))

    updated_state = sim_env.step(action)
    decision.decision_outcome(
        "DO_label",
        updated_state.reward,
        sight,
    )


def get_sight_instance():
  params = sight_pb2.Params(
      label='fractal_demo',
      bucket_name=f'{os.environ["PROJECT_ID"]}-sight',
  )
  sight_obj = Sight(params)
  return sight_obj


def main(argv: Sequence[str]) -> None:
  if len(argv) > 1:
    raise app.UsageError("Too many command-line arguments.")

  with get_sight_instance() as sight:
    decision.run(
      sight=sight,
      driver_fn=driver_fn,
      env=fourier_environment.get_fourier_rl_environment()
    )


if __name__ == "__main__":
  app.run(main)
