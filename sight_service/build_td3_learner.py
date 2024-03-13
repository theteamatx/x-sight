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

"""Setting up configuration for DQN Experiment."""

from acme.agents.jax import td3
from acme.jax import experiments


def build_td3_config():
  """Builds TD3 experiment config which can be executed in different ways."""

  def env_factory():
    # if env_name:
    #   return  wrappers.GymWrapper(gym.make(env_name))
    # else:
      return None

  network_factory = (
      lambda spec: td3.make_networks(spec, hidden_layer_sizes=(256, 256, 256)))

  # Construct the agent.
  config = td3.TD3Config(
      policy_learning_rate=3e-4,
      critic_learning_rate=3e-4,
  )
  td3_builder = td3.TD3Builder(config)

  return experiments.ExperimentConfig(
    builder=td3_builder,
    environment_factory=env_factory,
    network_factory=network_factory,
    seed=0,
    max_num_actor_steps=10)

