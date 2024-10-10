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

from absl import flags
from acme import specs
from acme import wrappers
from acme.agents.jax import d4pg
from acme.agents.jax.dqn import losses
from acme.jax import experiments
from acme.jax import networks as networks_lib
from acme.jax import utils
import gym
import haiku as hk

# SEED = flags.DEFINE_integer('seed', 0, 'Random seed.')
# NUM_STEPS = flags.DEFINE_integer(
#     'num_steps', 10, 'Number of env steps to run.'  # 1_000_000
# )


def build_d4pg_config():
  """Builds D4PG experiment config which can be executed in different ways."""

  def env_factory():
    # if env_name:
    #   return  wrappers.GymWrapper(gym.make(env_name))
    # else:
    return None

  vmax_values = {
      'gym': 1000.,
      'control': 150.,
  }
  vmax = vmax_values['gym']

  def network_factory(spec) -> d4pg.D4PGNetworks:
    return d4pg.make_networks(
        spec,
        policy_layer_sizes=(256, 256, 256),
        critic_layer_sizes=(256, 256, 256),
        vmin=-vmax,
        vmax=vmax,
    )

  # Configure the agent.
  d4pg_config = d4pg.D4PGConfig(learning_rate=3e-4, sigma=0.2)

  d4pg_builder = d4pg.D4PGBuilder(d4pg_config)

  return experiments.ExperimentConfig(
      builder=d4pg_builder,
      environment_factory=env_factory,
      network_factory=network_factory,
      seed=0,
      max_num_actor_steps=10,
  )
