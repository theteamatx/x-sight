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

import random
import dm_env
from acme import core
from acme import types

class SweetnessAgentAcme(core.Actor):
    
  def __init__(self):
    """Initializes the Sweetness environement."""

  def select_action(self, observation: types.NestedArray) -> types.NestedArray:
    """Samples from the policy and returns an action."""
    sweetness = random.randrange(0, 10)
    return sweetness

  def observe_first(self, timestep: dm_env.TimeStep):
    """Make a first observation from the environment.
    Note that this need not be an initial state, it is merely beginning the
    recording of a trajectory.
    Args:
      timestep: first timestep.
    """
    pass

  def observe(
      self,
      action: types.NestedArray,
      next_timestep: dm_env.TimeStep,
  ):
    """Make an observation of timestep data from the environment.
    Args:
      action: action taken in the environment.
      next_timestep: timestep produced by the environment given the action.
    """
    pass

  def update(self, wait: bool = False):
    """Perform an update of the actor parameters from past observations.
    Args:
      wait: if True, the update will be blocking.
    """
    pass