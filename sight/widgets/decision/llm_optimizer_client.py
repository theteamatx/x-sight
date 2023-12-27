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

"""Client for LLM optimizer to communicate with server."""

from typing import Optional, Sequence, Tuple

from absl import logging
from sight import service
from sight.proto import sight_pb2
from sight.widgets.decision.optimizer_client import OptimizerClient
from overrides import override

class LLMOptimizerClient (OptimizerClient):
  """LLM client for the Sight service."""

  def __init__(self, description: str, sight):
    super().__init__(sight_pb2.DecisionConfigurationStart.OptimizerType.OT_LLM) 
    self._description = description
    self._sight = sight
    self._worker_id = None
  
  @override
  def create_config(self) -> sight_pb2.DecisionConfigurationStart.ChoiceConfig:
    choice_config = sight_pb2.DecisionConfigurationStart.ChoiceConfig(
    )
    llm_config = sight_pb2.DecisionConfigurationStart.LLMConfig(
        description=self._description
      )
    choice_config.llm_config.CopyFrom(llm_config)
    return choice_config

  @override
  def decision_point(self, sight, request):
    for key, value in sight.widget_decision_state["state"].items():
      logging.info('key=%s / value=%s', key, value)
      param = request.decision_point.state_params.add()
      param.key = key
      param.value.sub_type
      param.value.double_value = value
    logging.info('request=%s', request)

    response = service.call(
        lambda s, meta: s.DecisionPoint(request, 300, metadata=meta)
    )
    return response.action[0].value.double_value

  @override
  def finalize_episode(self, sight, request):
    response = service.call(
        lambda s, meta: s.FinalizeEpisode(request, 300, metadata=meta)
    )
    return response