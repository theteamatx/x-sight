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

"""A sweetness RL environment built on the `dm_env.Environment` class.

Collection of RL environment which use discrete with action scaling/ continuous
and discrete without action scaling action specifications.
"""

from typing import Any

import dm_env
import numpy as np


class SweetnessEnvAcme(dm_env.Environment):
  """A Sweetness environment built on the `dm_env.Environment`.

  The inheriting class needs to implement the action specification.
  """

  def __init__(self):
    """Initializes the Sweetness environement.

    Args:
    """
    self._discount = 1.0
    self._current_step = 0

  def observation_spec(self) -> dm_env.specs.Array:
    """Returns the observation spec."""
    return dm_env.specs.Array(
        shape=(),
        dtype=np.int32,
        name="observation")

  def action_spec(self):
    """Defines the actions that should be provided to `step`.
    May use a subclass of `specs.Array` that specifies additional properties
    such as min and max bounds on the values.
    Returns:
      An `Array` spec, or a nested dict, list or tuple of `Array` specs.
    """
    return dm_env.specs.Array(
        shape=(),
        dtype=np.int32,
        name="action")

  def _observation(self) -> np.ndarray:
    """Returns observation array from base beer game environment."""
    return np.array(self._sweetooh)

  def reset(self) -> dm_env.TimeStep:
    """Returns the first `TimeStep` of a new episode."""

    self._discount = 1.0
    self._current_step = 0
    self._sweetooh = 100
    self._current_time_step = None

    return dm_env.restart(self._observation())

  def step(self, action) -> dm_env.TimeStep:
    """Updates the environment according to the action.

    Args:
      action:  A NumPy array, or a nested dict, list or tuple of arrays
        corresponding to `action_spec()`.

    Returns:
      acme environment timestep after taking the action.
    """
    # print('*'*30)
    joy = (action / 10) * self._sweetooh
    # print("joy : ", joy)
    reward = joy
    self._sweetooh -= joy
    # print("sweetooh : ", self._sweetooh)

    self._current_step += 1
    # print("_current_step : ",self._current_step)
    if self._current_step >= 10:
      return dm_env.termination(reward=reward, observation=self._observation())

    return dm_env.transition(
        reward=reward,
        observation=self._observation(),
        discount=self._discount)