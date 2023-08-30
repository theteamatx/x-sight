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

from absl import app
import functools
from acme import wrappers
from acme.agents.tf import dqn
from acme.tf import networks
import acme
import dm_env
import gym

def make_environment(evaluation: bool = False,
                     level: str = 'CartPole-v0',#'SpaceInvadersNoFrameskip-v4',
                     oar_wrapper: bool = False) -> dm_env.Environment:
    env = gym.make(level)#, full_action_space=True)
    make_episode_len = 108_000 if evaluation else 50_000

    wrapper_list = [
            wrappers.GymAtariAdapter,
            functools.partial(
                wrappers.AtariWrapper,
                to_float=True,
                # max_episode_len=max_episode_len,
                zero_discount_on_life_loss=True,
            )
    ]
    wrapper_list.append(wrappers.SinglePrecisionWrapper)
    return wrappers.wrap_all(env, wrapper_list)


def main(_):
    env = make_environment()

    env_spec = acme.make_environment_spec(env)

    network = networks.DQNAtariNetwork(env_spec.action.num_values)

    agent = dqn.DQN(env_spec, network)

    env_loop = acme.EnvironmentLoop(env, agent)
    env_loop.run(num_episode=10)

if __name__ == '__main__':
    app.run(main)
