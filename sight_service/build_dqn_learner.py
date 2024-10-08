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
from acme.agents.jax import dqn
from acme.agents.jax.dqn import losses
from acme.jax import experiments
from acme.jax import networks as networks_lib
from acme.jax import utils
import gym
import haiku as hk


def build_dqn_config():
  """Builds DQN experiment config which can be executed in different ways."""

  def env_factory():
    # if(env_name):
    #   return wrappers.GymWrapper(gym.make(env_name))
    # else:
    return None

  def net_factory(environment_spec: specs.EnvironmentSpec) -> dqn.DQNNetworks:
    """Creates networks for training DQN."""

    def network(inputs):
      # if(env_name):
      model = hk.Sequential([
          hk.nets.MLP([512, 128, environment_spec.actions.num_values]),
      ])
      # else:
      #   model = hk.Sequential([
      #       hk.nets.MLP([512, 128, possible_action_values]),
      #   ])
      return model(inputs)

    network_hk = hk.without_apply_rng(hk.transform(network))
    obs = utils.add_batch_dim(utils.zeros_like(environment_spec.observations))
    network = networks_lib.FeedForwardNetwork(
        init=lambda rng: network_hk.init(rng, obs), apply=network_hk.apply)
    typed_network = networks_lib.non_stochastic_network_to_typed(network)
    return dqn.DQNNetworks(policy_network=typed_network)

  config = dqn.DQNConfig(
      discount=0.99,
      learning_rate=1e-3,
      n_step=1,
      epsilon=0.1,
      min_replay_size=20,
      max_replay_size=1_000_000,
      batch_size=32,  # 256
      samples_per_insert=1,  # 0.5
  )

  loss_fn = losses.QLearning(discount=config.discount, max_abs_reward=1.0)

  dqn_builder = dqn.DQNBuilder(config, loss_fn=loss_fn)

  return experiments.ExperimentConfig(
      builder=dqn_builder,
      environment_factory=env_factory,
      network_factory=net_factory,
      seed=0,
      max_num_actor_steps=10,
  )
