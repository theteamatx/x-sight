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

from helpers.logs.logs_handler import logger as logging

import gym
import numpy as np
import dm_env
import math
from acme import wrappers
from sight import data_structures
from sight.widgets.decision import decision
from gym.utils import seeding
from acme import specs
import tree

_file_name = "driver.py"

# cartpole_env = wrappers.GymWrapper(CartPoleEnv())
# cartpole_env = wrappers.GymWrapper(gym.make('CartPole-v1'))
reset_next_step = True
gravity = 9.8
masscart = 1.0
masspole = 0.1
total_mass = masspole + masscart
length = 0.5  # actually half the pole's length
polemass_length = masspole * length
force_mag = 10.0
tau = 0.02  # seconds between state updates
kinematics_integrator = "euler"

# Angle at which to fail the episode
theta_threshold_radians = 12 * 2 * math.pi / 360
x_threshold = 2.4

state = None


def reset():
    global reset_next_step
    global state
    reset_next_step = False
    np_random, seed = seeding.np_random()
    low = -0.05
    high = 0.05
    state = np_random.uniform(low=low, high=high, size=(4, ))
    observation = np.array(state, dtype=np.float32)
    return dm_env.restart(observation)


def step(action):
    global reset_next_step
    global state
    if reset_next_step:
        return reset()

    # observation, reward, done, info = self._environment.step(action)

    x, x_dot, theta, theta_dot = state
    force = force_mag if action == 1 else -force_mag
    costheta = math.cos(theta)
    sintheta = math.sin(theta)

    temp = (force + polemass_length * theta_dot**2 * sintheta) / total_mass
    thetaacc = (gravity * sintheta - costheta * temp) / (
        length * (4.0 / 3.0 - masspole * costheta**2 / total_mass))
    xacc = temp - polemass_length * thetaacc * costheta / total_mass

    x = x + tau * x_dot
    x_dot = x_dot + tau * xacc
    theta = theta + tau * theta_dot
    theta_dot = theta_dot + tau * thetaacc
    state = (x, x_dot, theta, theta_dot)

    terminated = bool(x < -x_threshold or x > x_threshold
                      or theta < -theta_threshold_radians
                      or theta > theta_threshold_radians)
    if not terminated:
        reward = 1.0
    else:
        reward = 0.0

    observation, reward, done, info = np.array(
        state, dtype=np.float32), reward, terminated, {}
    reset_next_step = done

    # Convert the type of the reward based on the spec, respecting the scalar or
    # array property.
    reward = tree.map_structure(
        lambda x, t: (  # pylint: disable=g-long-lambda
            t.dtype.type(x)
            if np.isscalar(x) else np.asarray(x, dtype=t.dtype)),
        reward,
        specs.Array(shape=(), dtype=float, name='reward'))

    if done:
        truncated = info.get('TimeLimit.truncated', False)
        if truncated:
            return dm_env.truncation(reward, observation)
        return dm_env.termination(reward, observation)
    return dm_env.transition(reward, observation)


def driver_fn(sight) -> None:
    """Executes the logic of searching for a value.

  Args:
    env: The dm_env type env obcject used to call the reset and step methods.
    sight: The Sight logger object used to drive decisions.
  """
    method_name = 'driver_fn'
    logging.debug('>>>>>>>>>  In %s of %s', method_name, _file_name)

    timestep = reset()

    state_attrs = decision.get_state_attrs(sight)
    for i in range(len(state_attrs)):
        data_structures.log_var(state_attrs[i], timestep.observation[i], sight)

    while not timestep.last():
        chosen_action = decision.decision_point("DP_label", sight)
        timestep = step(chosen_action)

        for i in range(len(state_attrs)):
            data_structures.log_var(state_attrs[i], timestep.observation[i],
                                    sight)

        decision.decision_outcome(
            "DO_label",
            timestep.reward,
            sight,
        )
    logging.debug("<<<<  Out %s of %s", method_name, _file_name)
