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

"""Default Driver function to be used while training within the Sight log."""

import logging
import numpy as np
from sight import data_structures
# from sight.sight import Sight
from sight.widgets.decision import decision

_file_name = "driver.py"

def driver_fn(env, sight) -> None:
  """Executes the logic of searching for a value.

  Args:
    env: The dm_env type env obcject used to call the reset and step methods.
    sight: The Sight logger object used to drive decisions.
  """
  method_name = 'driver_fn'
  logging.debug('>>>>>>>>>  In %s of %s', method_name, _file_name)

  timestep = env.reset()
  # print('timestep : ', timestep)
  # raise SystemError

  state_attrs = decision.get_state_attrs(sight)
  for i in range(len(state_attrs)):
    data_structures.log_var(state_attrs[i], timestep.observation[i], sight)

  while not timestep.last():
    chosen_action = decision.decision_point("DP_label", sight)
    # print("chosen_action : ", chosen_action, type(chosen_action), chosen_action.shape)
    # raise SystemError

    # action_received = np.array(chosen_action["action_1"], dtype=np.int64)
    # todo : meetashah - dtype changed from np.int64 to np.float64 and added ndmin for d4pg
    # action_received = np.array(chosen_action["action_1"], dtype=np.float32, ndmin=1)
    # action_received = chosen_action[0]

    # timestep = env.step(action_received)
    timestep = env.step(chosen_action)
    # print("timestep :", timestep)

    for i in range(len(state_attrs)):
      data_structures.log_var(state_attrs[i], timestep.observation[i], sight)

    decision.decision_outcome(
        "DO_label",
        timestep.reward,
        sight,
    )
  logging.debug("<<<<  Out %s of %s", method_name, _file_name)
