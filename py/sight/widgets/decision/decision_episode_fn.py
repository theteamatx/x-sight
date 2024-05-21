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
"""Class that encapsulates the computation of a Sight Decision API episode."""

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Tuple

from absl import logging

import numpy as np
# import tensorflow as tf  # pylint: disable=g-explicit-tensorflow-version-import
# from tf_agents.specs import array_spec
# from tf_agents.specs import tensor_spec
# from tf_agents.trajectories import time_step as ts
from sight.proto import sight_pb2


@dataclass
class DecisionEpisodeFn:
    """The computation of a Sight Decision API episode and its meta-data."""

    # The function that runs a single episode.
    driver_fn: Callable[[Any], Any]

    # The names of the episode's state variables.
    state_attrs: List[str]

    # Maps the name of each state variable to its index in state_attrs.
    state_attr_to_idx: Dict[str, int]

    # Mapping from all the state variables to their minimum and maximum values.
    state_min: Dict[str, float]
    state_max: Dict[str, float]

    # datatype of state attrs
    # state_dtype: None

    # The TFAgents schema of the observation space.
    # observation_spec: array_spec.BoundedArraySpec

    # The names of the episode's action variables.
    action_attrs: List[str]

    # Maps the name of each action variable to its index in action_attrs.
    action_attr_to_idx: Dict[str, int]

    # Mapping from all the action variables to their minimum and maximum values.
    action_min: Dict[str, float]
    action_max: Dict[str, float]

    # datatype of action attrs
    # action_dtype: None

    # possible valid values of action attrs
    valid_action_values: Dict[str, int]

    # possible valid values of action attrs
    step_size: Dict[str, int]

    # The TFAgents schema of the action space.
    # action_spec: array_spec.BoundedArraySpec

    # The TFAgents schema of the rewards space.
    # reward_spec: array_spec.ArraySpec

    # The TFAgents schema of the time steps space (includes observations and
    # rewards).
    # time_step_spec: ts.TimeStep

    def __init__(
        self,
        driver_fn: Callable[[Any], Any],
        state_attrs: Dict[str, sight_pb2.DecisionConfigurationStart.AttrProps],
        action_attrs: Dict[str,
                           sight_pb2.DecisionConfigurationStart.AttrProps],
    ):
        self.driver_fn = driver_fn

        self.state_attrs = list(state_attrs.keys())
        self.state_attr_to_idx = {}
        for i in range(len(self.state_attrs)):
            self.state_attr_to_idx[self.state_attrs[i]] = i
        self.state_min = {
            attr: min_max.min_value
            for attr, min_max in state_attrs.items()
        }
        self.state_max = {
            attr: min_max.max_value
            for attr, min_max in state_attrs.items()
        }

        # for attr, val in state_attrs.items():
        #   self.state_dtype = val.datatype
        #   break

        # self.observation_spec = array_spec.BoundedArraySpec(
        #     shape=(len(state_attrs),),
        #     dtype=np.float32,
        #     minimum=[0] * len(state_attrs),
        #     maximum=[1] * len(state_attrs),
        #     name='observation',
        # )

        self.action_attrs = list(action_attrs.keys())
        self.action_attr_to_idx = {}
        for i in range(len(self.action_attrs)):
            self.action_attr_to_idx[self.action_attrs[i]] = i
        self.action_min = {
            attr: min_max.min_value
            for attr, min_max in action_attrs.items()
        }
        self.action_max = {
            attr: min_max.max_value
            for attr, min_max in action_attrs.items()
        }

        self.valid_action_values = {
            attr: attr_val.valid_int_values
            for attr, attr_val in action_attrs.items()
            if attr_val.valid_int_values
        }

        self.step_size = {
            attr: attr_val.step_size
            for attr, attr_val in action_attrs.items()
            if attr_val.step_size
        }

        # for action, attributes in action_attrs.items():
        #   if (attributes.valid_int_values):
        #     self.valid_action_values = attributes.valid_int_values

        # for attr, val in action_attrs.items():
        #   self.action_dtype = val.datatype
        #   break

        # if len(self.action_attrs) == 1:
        #   self.action_spec = array_spec.BoundedArraySpec(
        #       shape=(),
        #       dtype=np.float32,
        #       minimum=0,
        #       maximum=20,
        #       name='action',
        #   )
        # else:
        #   self.action_spec = array_spec.BoundedArraySpec(
        #       shape=(len(action_attrs),),
        #       dtype=np.float32,
        #       minimum=[0] * len(action_attrs),
        #       maximum=[20] * len(action_attrs),
        #       name='action',
        #   )

        # self.reward_spec = array_spec.ArraySpec(
        #     shape=(), dtype=np.float32, name='reward')
        # self.time_step_spec = ts.time_step_spec(self.observation_spec,
        #                                         self.reward_spec)

    # def tf_observation_spec(self) -> tf.TensorSpec:
    #   """Returns the TFAgents Tensor schema of the observation space."""
    #   return tensor_spec.from_spec(self.observation_spec)

    # def tf_action_spec(self) -> tf.TensorSpec:
    #   """Returns the TFAgents Tensor schema of the action space."""
    #   return tensor_spec.from_spec(self.action_spec)

    # def tf_time_step_spec(self) -> tf.TensorSpec:
    #   """Returns the TFAgents Tensor schema of the time step space."""
    #   return tensor_spec.from_spec(self.time_step_spec)

    # def create_tf_observation(self, state: Dict[str, float]) -> np.ndarray:
    #   """Creates an observation vector from a state dict.

    #   The values of the observation vector are in a canonical order and in the
    #   0-1 range, whereas the values in the state dict are in the range specified
    #   by the user when this object was initialized.

    #   Args:
    #     state: Maps each state attribute to its value.

    #   Returns:
    #     The array that contains state attribute values, ordered as in
    #     self.state_attrs.
    #   """
    #   return np.array([[(state[v] - self.state_min[v] /
    #                      (self.state_max[v] - self.state_min[v]))
    #                     for v in self.state_attrs]],
    #                   dtype=np.float32)

    def create_user_action(self, action: np.ndarray) -> Dict[str, float]:
        """Converts an action vector to a dictionary of actions for the user.

    The actions dictionary maps the name of each action to its value in the
    range specified by the user when this object was initialized. The values
    in the actions vector are in the 0-1 range.

    Args:
      action: Maps each action attribute to its value.

    Returns:
      The array that contains action attribute values, ordered as in
      self.action_attrs.
    """
        if len(self.action_attrs) == 1:
            action_list = [action.numpy()]
        else:
            action_list = list(action[0].numpy())
        action_dict = {}
        for i in range(len(action_list)):
            var = self.action_attrs[i]
            action_dict[var] = (action_list[i] / 20) * (
                self.action_max[var] -
                self.action_min[var]) + self.action_min[var]

        return action_dict

    def run(self, sight: Any) -> None:
        """Generates a dataset for a single episode."""
        self.driver_fn(sight)
