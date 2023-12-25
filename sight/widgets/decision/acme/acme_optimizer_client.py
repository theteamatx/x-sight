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

"""Client for dm-acme optimizer to communicate with server."""
import math
import logging
from typing import Optional, Sequence, Tuple
from acme import specs
from acme.jax.experiments import config
import dm_env
import jax
import numpy as np
import reverb
from sight.widgets.decision.acme import sight_adder
from sight.widgets.decision.acme import sight_variable_source
from sight.widgets.decision.acme.build_actor import build_actor_config
from sight.widgets.decision.optimizer_client import OptimizerClient
from overrides import override

_file_name = "acme_optimizer_client.py"


def generate_spec_details(attr_dict):
  """convert the spec details of environment into usable format."""
  method_name = "generate_spec_details"
  logging.debug(">>>>  In %s of %s", method_name, _file_name)
  state_min = np.array(list(attr_dict.state_min.values()))
  state_max = np.array(list(attr_dict.state_max.values()))
  state_param_length = len(attr_dict.state_attrs)
  # # for only 1 action
  # action_min = list(attr_dict.action_min.values())[0]
  # action_max = list(attr_dict.action_max.values())[0]
  action_min = np.array(list(attr_dict.action_min.values()))
  action_max = np.array(list(attr_dict.action_max.values()))
  action_param_length = len(attr_dict.action_attrs)
  possible_actions = 1
  for action_attr in attr_dict.action_attrs:
    possible_actions *= int(
        attr_dict.action_max[action_attr]
        - attr_dict.action_min[action_attr]
        + 1
    )

  logging.debug("<<<<  Out %s of %s", method_name, _file_name)
  return (
      state_min,
      state_max,
      state_param_length,
      action_min,
      action_max,
      action_param_length,
      possible_actions,
  )


class AcmeOptimizerClient (OptimizerClient):
  """Acme client for the Sight service."""

  def __init__(self, sight):
    super().__init__(sight_pb2.DecisionConfigurationStart.OptimizerType.OT_ACME) 
    self._sight = sight
    self._actor = None
    self._adder = None
    self._variable_source = None
    self._dp_first_call = True
    self._last_acme_action = None

    # added to run the base example
    self._replay_server = None
    self._replay_client = None
    self._dataset = None
    self._learner = None
  
  @override
  def create_config(self) -> sight_pb2.DecisionConfigurationStart.ChoiceConfig:
    choice_config = sight_pb2.DecisionConfigurationStart.ChoiceConfig()
    (
        state_min,
        state_max,
        state_param_length,
        action_min,
        action_max,
        action_param_length,
        possible_actions,
    ) = self.generate_spec_details(
        self._sight.widget_decision_state['decision_episode_fn']
    )
    choice_config.acme_config.state_min.extend(state_min)
    choice_config.acme_config.state_max.extend(state_max)
    choice_config.acme_config.state_param_length = state_param_length
    choice_config.acme_config.action_min.extend(action_min)
    choice_config.acme_config.action_max.extend(action_max)
    choice_config.acme_config.action_param_length = action_param_length
    choice_config.acme_config.possible_actions = possible_actions
    return choice_config

  def generate_env_spec(
      self,
      state_min,
      state_max,
      state_param_length,
      action_min,
      action_max,
      action_param_length,
  ) -> specs.EnvironmentSpec:
    """Generates the environment spec for the environment."""

    method_name = "generate_env_spec"
    logging.debug(">>>>  In %s of %s", method_name, _file_name)

    new_env_spec = specs.EnvironmentSpec(
        # works for gym
        # observations=specs.BoundedArray(
        #     shape=(state_param_length,),
        #     dtype=np.float32,
        #     name="observation",
        #     minimum=state_min,
        #     maximum=state_max,
        # ),
        # ? works for fractal states
        observations=specs.Array(
            shape=(state_param_length,),
            dtype=np.float32,
            name="observation"
        ),

        # ? works for only 1 action parameter
        # actions=specs.BoundedArray(
        #     shape=(),
        #     dtype=int,
        #     name="action",
        #     minimum=action_min[0],
        #     maximum=action_max[0],
        # ),

        actions=specs.BoundedArray(
            shape=(action_param_length,),
            dtype=int,
            name="action",
            minimum=action_min[0],
            maximum=action_max[0],
        ),
        rewards=specs.Array(shape=(), dtype=float, name="reward"),
        discounts=specs.BoundedArray(
            shape=(), dtype=float, minimum=0.0, maximum=1.0, name="discount"
        ),
    )
    logging.debug("<<<<  Out %s of %s", method_name, _file_name)
    return new_env_spec

  def create_new_actor(self):
    """Creates a new actor."""
    method_name = "create_new_actor"
    logging.debug(">>>>  In %s of %s", method_name, _file_name)

    (
        state_min,
        state_max,
        state_param_length,
        action_min,
        action_max,
        action_param_length,
        possible_actions,
    ) = generate_spec_details(
        self._sight.widget_decision_state['decision_episode_fn']
    )
    print('possible_actions : ', possible_actions)
    experiment = build_actor_config(possible_action_values=possible_actions)
    environment_spec = self.generate_env_spec(
        state_min,
        state_max,
        state_param_length,
        action_min,
        action_max,
        action_param_length,
    )

    networks = experiment.network_factory(environment_spec)
    policy = config.make_policy(
        experiment=experiment,
        networks=networks,
        environment_spec=environment_spec,
        evaluation=False,
    )
    self._adder = sight_adder.SightAdder()
    self._variable_source = sight_variable_source.SightVariableSource(
        adder=self._adder, client_id=self._sight.id, sight=self._sight
    )


    key = jax.random.PRNGKey(0)
    actor_key, key = jax.random.split(key)
    logging.debug("<<<<  Out %s of %s", method_name, _file_name)
    return experiment.builder.make_actor(
        actor_key,
        policy,
        environment_spec,
        variable_source=self._variable_source,
        adder=self._adder,
    )

  @override
  def decision_point(self, sight):
    """communicates with decision_point method on server.

    Stores the trajectories locally, after storing 50 trajectories, calls
    Update on actor so send those to server and fetch latest weights from
    learner.

    Args:
      sight: sight object.
    Returns:
      action to be performed.
    """
    method_name = "decision_point"
    logging.debug(">>>>  In %s of %s", method_name, _file_name)


    observation = np.array(
        list(sight.widget_decision_state["state"].values()),
        dtype=np.float32,
    )
    # print('observation : ', observation)
    if self._dp_first_call:
      # create actor, if not there
      if self._actor is None:
        print("no actor found, creating new one.....")
        self._actor = self.create_new_actor()
        # update will fetch the latest weights from learner into actor policy
        self._actor.update(wait=True)

      timestep = dm_env.TimeStep(
          step_type=dm_env.StepType.FIRST,
          reward=None,
          discount=None,
          observation=np.array(observation),
      )
      self._actor.observe_first(timestep)
      self._dp_first_call = False
    else:
      # do this for subsequent call
      timestep = dm_env.TimeStep(
          step_type=dm_env.StepType.MID,
          reward=np.array(
              sight.widget_decision_state["outcome_value"], dtype=np.float64
          ),
          discount=np.array(
              sight.widget_decision_state["discount"], dtype=np.float64
          ),
          observation=np.array(observation, dtype=np.float32),
      )
      action = np.array(self._last_acme_action, dtype=np.int64)
      self._actor.observe(action, next_timestep=timestep)

      if len(self._actor._adder._observation_list) % 50 == 0:
        self._actor.update(wait=True)

    # print('action : ', self._actor.select_action(observation))
    self._last_acme_action = float(self._actor.select_action(observation))
    logging.debug("<<<<  Out %s of %s", method_name, _file_name)
    return self._last_acme_action

  @override
  def finalize_episode(self, sight):
    """completes episode and stores remaining local trajectories to server.

    Args:
      sight: sight object.
    """
    method_name = "finalize_episode"
    logging.debug(">>>>  In %s of %s", method_name, _file_name)

    observation = np.array(
        list(sight.widget_decision_state["state"].values()),
        dtype=np.float32,
    )
    timestep = dm_env.TimeStep(
        step_type=dm_env.StepType.LAST,
        reward=np.array(
            sight.widget_decision_state["outcome_value"], dtype=np.float64
        ),
        discount=np.array(
            sight.widget_decision_state["discount"], dtype=np.float64
        ),
        observation=np.array(observation, dtype=np.float32),
    )
    # action = np.array([self._last_acme_action], dtype=np.int64)
    action = np.array(self._last_acme_action, dtype=np.int64)
    self._actor.observe(action, next_timestep=timestep)

    # send remaining records to server and fetch latest weights in response
    # if len(self._actor._adder._observation_list) % 50 == 0:
    self._actor.update(wait=True)
    # self._actor._adder.reset()  # _actor._adder._observation_list = []

    # resetting this global varibale so, next iteration will
    # start with observer_first
    self._dp_first_call = True

    logging.debug("<<<<  Out %s of %s", method_name, _file_name)
