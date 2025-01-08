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
"""An instance of a Sight optimizer dedicated to a single experiment."""

from concurrent import futures
from typing import Any, Dict, List, Sequence, Tuple

from helpers.logs.logs_handler import logger as logging
from sight.proto import sight_pb2
from sight_service import utils
from sight_service.proto import service_pb2

_file_name = "optimizer_instance.py"


class OptimizerInstance:
  """An OptimizerInstance class that is generic for all optimizers.

  An optimizer containing base methods which specialized optimizers will
  override while communicating with client.
  """

  def __init__(self):
    self.num_trials = 0
    self.actions = {}
    self.state = {}
    self.outcomes = {}

  def launch(self,
             request: service_pb2.LaunchRequest) -> service_pb2.LaunchResponse:
    """Initializing new study and storing state and action attributes for the same.
    """
    method_name = "launch"
    logging.debug(">>>>  In %s of %s", method_name, _file_name)
    # logging.info('request.decision_config_params=%s', request.decision_config_params)

    self.num_trials = request.decision_config_params.num_trials

    # sorting dict key wise to maintain consistency at for all calls
    action_keys = list(request.decision_config_params.action_attrs.keys())
    action_keys.sort()
    for k in action_keys:
      self.actions[k] = request.decision_config_params.action_attrs[k]

    # sorting dict key wise to maintain consistency at for all calls
    state_keys = list(request.decision_config_params.state_attrs.keys())
    state_keys.sort()
    for k in state_keys:
      self.state[k] = request.decision_config_params.state_attrs[k]

    # sorting dict key wise to maintain consistency at for all calls
    outcome_keys = list(request.decision_config_params.outcome_attrs.keys())
    outcome_keys.sort()
    for k in outcome_keys:
      self.outcomes[k] = request.decision_config_params.outcome_attrs[k]

    # print(f"<<<<<<<<<  Out {method_name} of {_file_name}.")
    logging.debug("<<<<  Out %s of %s", method_name, _file_name)
    return service_pb2.LaunchResponse()

  def decision_point(
      self, request: service_pb2.DecisionPointRequest
  ) -> service_pb2.DecisionPointResponse:
    return service_pb2.DecisionPointResponse()

  def finalize_episode(
      self, request: service_pb2.FinalizeEpisodeRequest
  ) -> service_pb2.FinalizeEpisodeResponse:
    return service_pb2.FinalizeEpisodeResponse()

  def tell(self, request: service_pb2.TellRequest) -> service_pb2.TellResponse:
    return service_pb2.TellResponse()

  def listen(self,
             request: service_pb2.ListenRequest) -> service_pb2.ListenResponse:
    return service_pb2.ListenResponse()

  def current_status(
      self, request: service_pb2.CurrentStatusRequest
  ) -> service_pb2.CurrentStatusResponse:
    return service_pb2.CurrentStatusResponse()

  def propose_action(
      self, request: service_pb2.ProposeActionRequest
  ) -> service_pb2.ProposeActionResponse:
    return service_pb2.ProposeActionResponse()

  def GetOutcome(
      self,
      request: service_pb2.GetOutcomeRequest) -> service_pb2.GetOutcomeResponse:
    return service_pb2.GetOutcomeResponse()

  def fetch_optimal_action(
      self, request: service_pb2.FetchOptimalActionRequest
  ) -> service_pb2.FetchOptimalActionResponse:
    return service_pb2.FetchOptimalActionResponse()

  def close(self,
            request: service_pb2.CloseRequest) -> service_pb2.CloseResponse:
    return service_pb2.CloseResponse()

  def WorkerAlive(
      self, request: service_pb2.WorkerAliveRequest
  ) -> service_pb2.WorkerAliveResponse:
    return service_pb2.WorkerAliveResponse()
