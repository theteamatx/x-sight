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

reset_next_step = True
state = None
elapsed_steps = 0
max_episode_steps = 200

DEFAULT_X = np.pi
DEFAULT_Y = 1.0

max_speed = 8
max_torque = 2.0
dt = 0.05
g = 10.0
m = 1.0
l = 1.0


def reset():
    global reset_next_step
    global state
    global elapsed_steps
    elapsed_steps = 0
    reset_next_step = False
    np_random, seed = seeding.np_random()
    high = np.array([DEFAULT_X, DEFAULT_Y])
    low = -high
    state = np_random.uniform(low=low, high=high)
    theta, thetadot = state
    observation = np.array(
        [np.cos(theta), np.sin(theta), thetadot], dtype=np.float32)
    return dm_env.restart(observation)


def step(action):
    # from gym_wrapper.py
    global reset_next_step
    global state
    global elapsed_steps
    # print("State : ", state)
    if reset_next_step:
        return reset()

    # step of pendulum
    th, thdot = state
    u = np.clip(action, -max_torque, max_torque)[0]
    costs = angle_normalize(th)**2 + 0.1 * thdot**2 + 0.001 * (u**2)

    newthdot = thdot + (3 * g / (2 * l) * np.sin(th) + 3.0 /
                        (m * l**2) * u) * dt
    newthdot = np.clip(newthdot, -max_speed, max_speed)
    newth = th + newthdot * dt

    state = np.array([newth, newthdot])
    theta, thetadot = state
    latest_state = np.array(
        [np.cos(theta), np.sin(theta), thetadot], dtype=np.float32)

    observation, reward, done, info = latest_state, -costs, False, {}
    elapsed_steps += 1
    if elapsed_steps >= max_episode_steps:
        done = True

    # from gym_wrapper.py
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


def angle_normalize(x):
    return ((x + np.pi) % (2 * np.pi)) - np.pi


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
