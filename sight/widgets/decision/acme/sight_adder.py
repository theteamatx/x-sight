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

"""Custom implementation of base Adder."""

import logging
from typing import Any, Optional
from acme import types
from acme.adders import base
import dm_env
from service import service_pb2
from service.numproto.numproto import ndarray_to_proto

_file_name = "sight_adder.py"

class SightAdder(base.Adder):
  """A custom adder based on the base.Adder with some logic changes.

  This adder maintains observations provided via actor in a list.
  """

  def __init__(self):
    """Initialize a CustomAdder instance."""
    self._observation_list = []
    self._existing_batch_last_record = None

  def reset(self, timeout_ms: Optional[int] = None):
    """Resets the adder's buffer."""
    # reset called at initial stage or afrer whole episode completed
    if (
        not self._existing_batch_last_record
        or self._existing_batch_last_record["next_timestep"].last()
    ):
      self._observation_list = []
    # whole episode not completed so, converting last record of this batch
    # as FIRST type record for next batch
    else:
      timestep = dm_env.TimeStep(
          step_type=dm_env.StepType.FIRST,
          reward=None,
          discount=None,
          observation=self._existing_batch_last_record[
              "next_timestep"
          ].observation,
      )
      observation_dict = {"action": None, "next_timestep": timestep}
      self._observation_list = [observation_dict]

  def observation_to_proto(self, observation: dict[str, Any]):
    method_name = "observation_to_proto"
    logging.debug(">>>>  In %s of %s", method_name, _file_name)
    obs = service_pb2.Acme_Request().Observation()

    if observation["action"]:
      obs.action.CopyFrom(ndarray_to_proto(observation["action"]))
    obs.steptype = observation["next_timestep"].step_type
    if observation["next_timestep"].reward:
      obs.reward.CopyFrom(ndarray_to_proto(observation["next_timestep"].reward))
    if observation["next_timestep"].discount:
      obs.discount.CopyFrom(
          ndarray_to_proto(observation["next_timestep"].discount)
      )
    obs.observation.CopyFrom(
        ndarray_to_proto(observation["next_timestep"].observation)
    )
    logging.debug("<<<<  Out %s of %s", method_name, _file_name)
    return obs

  def fetch_and_reset_observation_list(self, sight_client_id, sight_worker_id):
    method_name = "fetch_and_reset_observation_list"
    logging.debug(">>>>  In %s of %s", method_name, _file_name)
    request = service_pb2.DecisionPointRequest()
    request.client_id = str(sight_client_id)
    request.worker_id = str(sight_worker_id)

    acme_config = service_pb2.Acme_Request()
    if len(self._observation_list) > 0:
      for episode_obs in self._observation_list:
        obs = self.observation_to_proto(episode_obs)
        acme_config.episode_observations.append(obs)

    request.acme_decision_point.CopyFrom(acme_config)
    self.reset()
    logging.debug("<<<<  Out %s of %s", method_name, _file_name)
    return request

  def add_first(self, timestep: dm_env.TimeStep):
    """Record the first observation of a trajectory."""
    self.add(action=None, next_timestep=timestep)

  def add(
      self,
      action: types.NestedArray,
      next_timestep: dm_env.TimeStep,
      extras: types.NestedArray = (),
  ):
    """Record an action and the following timestep."""
    observation_dict = {"action": action, "next_timestep": next_timestep}
    self._existing_batch_last_record = observation_dict
    self._observation_list.append(observation_dict)
