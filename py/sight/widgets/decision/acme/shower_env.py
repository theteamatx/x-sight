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
"""Custom implementation of manage shower temperature environment."""

import random

import dm_env
import numpy as np


class ShowerEnv(dm_env.Environment):
  """Custom environment for managing shower temperature."""

  def __init__(self):
    self.state = 38 + random.randint(-3, 3)
    self.shower_length = 60

  def action_spec(self):
    return dm_env.specs.BoundedArray(shape=(),
                                     dtype=int,
                                     name='action',
                                     minimum=0,
                                     maximum=2)

  def observation_spec(self):
    return dm_env.specs.BoundedArray(
        # shape=(1,),
        shape=(),
        dtype=np.float32,
        name='observation',
        minimum=0,
        maximum=100,
    )

  def step(self, action):
    # Apply action
    # 0 -1 = -1 temperature
    # 1 -1 = 0
    # 2 -1 = 1 temperature
    self.state += action - 1
    self.shower_length -= 1

    # Calculate reward
    if self.state >= 37 and self.state <= 39:
      reward = 1
    else:
      reward = -1

    # Check if shower is done
    if self.shower_length <= 0:
      done = True
    else:
      done = False

    # Return step information
    if done:
      return dm_env.termination(reward, np.array([self.state],
                                                 dtype=np.float32))
    else:
      return dm_env.transition(reward, np.array([self.state], dtype=np.float32))

  def render(self):
    pass

  def reset(self):
    # Reset shower temperature
    self.state = 38 + random.randint(-3, 3)
    self.shower_length = 60
    return dm_env.restart(np.array([self.state], dtype=np.float32))
