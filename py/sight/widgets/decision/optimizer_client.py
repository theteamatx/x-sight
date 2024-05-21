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

"""Base implementation of optimizer clients that communicate with the decision service."""

from typing import Any, Dict, Optional, Sequence, Tuple

from sight_service.proto import service_pb2
from sight import service_utils as service
from sight.proto import sight_pb2

class OptimizerClient:
  """Generic optimizer for the Sight Decision service."""

  def __init__(self, optimizer_type: sight_pb2.DecisionConfigurationStart.OptimizerType):
    self._optimizer_type = optimizer_type

  def optimizer_type(self) -> sight_pb2.DecisionConfigurationStart.OptimizerType:
    return self._optimizer_type

  def create_config(self) -> sight_pb2.DecisionConfigurationStart.ChoiceConfig:
    return sight_pb2.DecisionConfigurationStart.ChoiceConfig()

  def decision_point(self, sight, request: service_pb2.DecisionPointRequest):
    response = service.call(
        lambda s, meta: s.DecisionPoint(request, 300, metadata=meta)
    )

    return self._get_dp_action(response)

  def _get_dp_action(self, dp_response: service_pb2.DecisionPointResponse) -> Dict[str, float]:
    """Returns the dict representation of the action encoded in dp_response."""
    d = {}
    for a in dp_response.action:
      d[a.key] = a.value.double_value
    return d

  def _set_dp_action(self, dp: sight_pb2.DecisionPoint, action: Dict[str, float]) -> None:
    """Add to dp the attributes of action."""
    for key, val in action.items():
      dp.value.add(sight_pb2.DecisionParam(key=key, value=sight_pb2.Value(double_value=val)))

  def finalize_episode(self, sight, request: service_pb2.FinalizeEpisodeRequest):
    response = service.call(
        lambda s, meta: s.FinalizeEpisode(request, 300, metadata=meta)
    )
    return response
