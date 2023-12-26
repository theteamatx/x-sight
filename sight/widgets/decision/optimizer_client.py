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

from typing import Optional, Sequence, Tuple

from sight import service
from sight.proto import sight_pb2

class OptimizerClient:
  """Generic optimizer for the Sight Decision service."""

  def __init__(self, optimizer_type: sight_pb2.DecisionConfigurationStart.OptimizerType):
    self._optimizer_type = optimizer_type

  def optimizer_type(self) -> sight_pb2.DecisionConfigurationStart.OptimizerType:
    return self._optimizer_type

  def create_config(self) -> sight_pb2.DecisionConfigurationStart.ChoiceConfig:
    return sight_pb2.DecisionConfigurationStart.ChoiceConfig()

  def decision_point(self, sight, request):
    response = service.call(
        lambda s, meta: s.DecisionPoint(request, 300, metadata=meta)
    )
    print("response **************: ",response)

    return response.action[0].value.double_value

  def finalize_episode(self, sight, request):
    response = service.call(
        lambda s, meta: s.FinalizeEpisode(request, 300, metadata=meta)
    )
    return response