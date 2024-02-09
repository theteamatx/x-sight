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
from absl import flags
from acme import specs
from acme.jax.experiments import config
import dm_env
import jax
import numpy as np
import reverb
from sight_service.proto import service_pb2
from sight.proto import sight_pb2
from sight.widgets.decision.acme import sight_adder
from sight.widgets.decision.acme import sight_variable_source
from sight.widgets.decision.acme.build_dqn_actor import build_dqn_config
from sight.widgets.decision.acme.build_d4pg_actor import build_d4pg_config
from sight.widgets.decision.optimizer_client import OptimizerClient
from overrides import override

_ACME_AGENT = flags.DEFINE_enum(
    'acme_agent',
    None,
    ['dqn', 'd4pg'],
    'The acme provided jax agent to use',
)

FLAGS = flags.FLAGS
_file_name = "acme_optimizer_client.py"

# def fetch_possible_actions(attr_dict):
#   possible_actions = 1
#   for action_attr in attr_dict.action_attrs:
#     possible_actions *= int(
#         attr_dict.action_max[action_attr]
#         - attr_dict.action_min[action_attr]
#         + 1
#     )
#   return possible_actions


# def generate_spec_details(attr_dict):
#   """convert the spec details of environment into usable format."""
#   method_name = "generate_spec_details"
#   logging.debug(">>>>  In %s of %s", method_name, _file_name)
#   state_min = np.array(list(attr_dict.state_min.values()))
#   state_max = np.array(list(attr_dict.state_max.values()))
#   state_param_length = len(attr_dict.state_attrs)
#   # # for only 1 action
#   # action_min = list(attr_dict.action_min.values())[0]
#   # action_max = list(attr_dict.action_max.values())[0]
#   action_min = np.array(list(attr_dict.action_min.values()))
#   action_max = np.array(list(attr_dict.action_max.values()))
#   action_param_length = len(attr_dict.action_attrs)
#   # possible_actions = 1
#   # for action_attr in attr_dict.action_attrs:
#   #   possible_actions *= int(
#   #       attr_dict.action_max[action_attr]
#   #       - attr_dict.action_min[action_attr]
#   #       + 1
#   #   )

#   logging.debug("<<<<  Out %s of %s", method_name, _file_name)
#   return (
#       state_min,
#       state_max,
#       state_param_length,
#       action_min,
#       action_max,
#       action_param_length,
#       # possible_actions,
#   )


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
    # print("self._sight.widget_decision_state['decision_episode_fn'] : ", self._sight)
    print("in create config")
    choice_config = sight_pb2.DecisionConfigurationStart.ChoiceConfig()

    if(FLAGS.acme_agent == 'dqn'):
      choice_config.acme_config.acme_agent = sight_pb2.DecisionConfigurationStart.AcmeConfig.AA_DQN
    elif(FLAGS.acme_agent == 'd4pg'):
      choice_config.acme_config.acme_agent = sight_pb2.DecisionConfigurationStart.AcmeConfig.AA_D4PG

    # possible_actions = fetch_possible_actions(self._sight.widget_decision_state['decision_episode_fn'])
    # choice_config.acme_config.possible_actions = possible_actions

    #? using state and action related data as common to all choice_config
    # (
    #     state_min,
    #     state_max,
    #     state_param_length,
    #     action_min,
    #     action_max,
    #     action_param_length,
    #     possible_actions,
    # ) = generate_spec_details(
    #     self._sight.widget_decision_state['decision_episode_fn']
    # )
    # choice_config.acme_config.state_min.extend(state_min)
    # choice_config.acme_config.state_max.extend(state_max)
    # choice_config.acme_config.state_param_length = state_param_length
    # choice_config.acme_config.action_min.extend(action_min)
    # choice_config.acme_config.action_max.extend(action_max)
    # choice_config.acme_config.action_param_length = action_param_length
    # choice_config.acme_config.possible_actions = possible_actions

    # if FLAGS.env_name:
    #   choice_config.acme_config.env_name = FLAGS.env_name

    return choice_config

  def generate_env_spec(
      self,
      # state_min,
      # state_max,
      # state_param_length,
      # action_min,
      # action_max,
      # action_param_length,
      attr_dict
  )-> specs.EnvironmentSpec:
    """Generates the environment spec for the environment."""

    method_name = "generate_env_spec"
    logging.debug(">>>>  In %s of %s", method_name, _file_name)

    dtype_mapping = {
        sight_pb2.DecisionConfigurationStart.DataType.DT_INT32: np.int32,
        sight_pb2.DecisionConfigurationStart.DataType.DT_INT64: np.int64,
        sight_pb2.DecisionConfigurationStart.DataType.DT_FLOAT32: np.float32,
        sight_pb2.DecisionConfigurationStart.DataType.DT_FLOAT64: np.float64,
    }

    state_min = np.array(list(attr_dict.state_min.values()))
    state_max = np.array(list(attr_dict.state_max.values()))
    state_param_length = len(attr_dict.state_attrs)
    state_dtype = dtype_mapping[attr_dict.state_dtype]
    observations = specs.BoundedArray(
            shape=(state_param_length,),
            dtype=state_dtype,
            name="observation",
            minimum=state_min,
            maximum=state_max,
        ),

    action_min = np.array(list(attr_dict.action_min.values()))
    action_max = np.array(list(attr_dict.action_max.values()))
    action_param_length = len(attr_dict.action_attrs)
    action_dtype = dtype_mapping[attr_dict.action_dtype]

    # create discrete spec
    if(attr_dict.valid_action_values):
      actions = specs.DiscreteArray(
        num_values=attr_dict.valid_action_values,
        dtype=action_dtype,
        name="action",
      )
    # create bounded spec
    else:
      actions = specs.BoundedArray(
              shape=(action_param_length,),
              dtype=action_dtype,
              name="action",
              minimum=action_min,
              maximum=action_max,
          )

    # print(state_dtype, action_dtype)

    new_env_spec = specs.EnvironmentSpec(
        # works for gym
        observations=observations,
        actions=actions,
        rewards=specs.Array(shape=(), dtype=float, name="reward"),
        discounts=specs.BoundedArray(
            shape=(), dtype=float, minimum=0.0, maximum=1.0, name="discount"
        ),
    )
    logging.debug("<<<<  Out %s of %s", method_name, _file_name)
    # print("new_env_spec : ", new_env_spec)
    return new_env_spec

  def create_new_actor(self):
    """Creates a new actor."""
    method_name = "create_new_actor"
    logging.debug(">>>>  In %s of %s", method_name, _file_name)

    # if FLAGS.env_name:
    #   if FLAGS.env_name == "Pendulum-v1":
    #     experiment = build_d4pg_config(env_name=FLAGS.env_name)
    #   else:
    #     experiment = build_dqn_config(env_name=FLAGS.env_name)
    #   # print("experiment : ", experiment)

    #   environment = experiment.environment_factory()
    #   environment_spec = specs.make_environment_spec(environment)
    #   # print('environment_spec : ', environment_spec)

    # else:
    attr_dict = self._sight.widget_decision_state['decision_episode_fn']

    if(FLAGS.acme_agent == 'dqn'):
      possible_actions=attr_dict.valid_action_values
      experiment = build_dqn_config(possible_action_values=possible_actions)
    # elif(FLAGS.acme_agent == sight_pb2.DecisionConfigurationStart.AcmeConfig.AA_D4PG):
    else:
      experiment = build_d4pg_config()

    # (
    #   state_min,
    #   state_max,
    #   state_param_length,
    #   state_dtype,
    #   action_min,
    #   action_max,
    #   action_param_length,
    #   action_dtype
    #   # possible_actions,
    # ) = generate_spec_details(
    #     self._sight.widget_decision_state['decision_episode_fn']
    # )
    environment_spec = self.generate_env_spec(
        # state_min,
        # state_max,
        # state_param_length,
        # action_min,
        # action_max,
        # action_param_length,
      attr_dict
    )

    # print('environment_spec : ', environment_spec)

    networks = experiment.network_factory(environment_spec)
    policy = config.make_policy(
        experiment=experiment,
        networks=networks,
        environment_spec=environment_spec,
        evaluation=False,
    )
    # print("network : ", networks)
    # print("policy : ", policy)

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
  def decision_point(self, sight, request: service_pb2.DecisionPointRequest):
  # def decision_point(self, sight):
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
    # logging.info(">>>>  In %s of %s", method_name, _file_name)

    observation = np.array(
        list(sight.widget_decision_state["state"].values()),
        dtype=np.float32,
        # todo : meetashah - this should be extracted from env
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
          observation=observation,
      )
      self._actor.observe_first(timestep)
      self._dp_first_call = False
    else:
      # do this for subsequent call
      # logging.info("subsequent call of decision_point...")
      timestep = dm_env.TimeStep(
          step_type=dm_env.StepType.MID,
          reward=np.array(
              sight.widget_decision_state["outcome_value"], dtype=np.float64
          ),
          discount=np.array(
              sight.widget_decision_state["discount"], dtype=np.float64
          ),
          observation=observation,
      )

      # action = np.array(self._last_acme_action, dtype=np.int64)
      # todo : meetashah - changed dtyep from int64 to float32 for d4pg agent
      # action = np.array(self._last_acme_action, dtype=np.float32, ndmin=1)

      # self._actor.observe(action, next_timestep=timestep)
      self._actor.observe(self._last_acme_action, next_timestep=timestep)

      if len(self._actor._adder._observation_list) % 50 == 0:
        self._actor.update(wait=True)

    # store current action for next call as last_action
    self._last_acme_action = self._actor.select_action(observation)
    # print("last_Acme_Action : ", self._last_acme_action, self._last_acme_action.dtype, type(self._last_acme_action), self._last_acme_action.shape)
    # raise SystemError

    # todo:meetashah- for dqn-cartpole, we get dtype int32 but require int64
    if(self._last_acme_action.dtype == 'int32'):
      self._last_acme_action = np.array(self._last_acme_action, dtype=np.int64)
      # self._last_acme_action = self._last_acme_action.reshape((1,))


    # print("last_Acme_Action : ", self._last_acme_action, self._last_acme_action.dtype, self._last_acme_action.shape)
    # raise SystemError
    # logging.info("<<<<  Out %s of %s", method_name, _file_name)
    return self._last_acme_action

  @override
  def finalize_episode(self, sight, request: service_pb2.FinalizeEpisodeRequest):
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
    # action = np.array(self._last_acme_action, dtype=np.int64)
    # todo : meetashah - changed dtyep from int64 to float64 for d4pg agent
    # action = np.array(self._last_acme_action, dtype=np.float32)
    # self._actor.observe(action, next_timestep=timestep)
    self._actor.observe(self._last_acme_action, next_timestep=timestep)


    # send remaining records to server and fetch latest weights in response
    # if len(self._actor._adder._observation_list) % 50 == 0:
    self._actor.update(wait=True)
    # self._actor._adder.reset()  # _actor._adder._observation_list = []

    # resetting this global varibale so, next iteration will
    # start with observer_first
    self._dp_first_call = True

    logging.debug("<<<<  Out %s of %s", method_name, _file_name)
