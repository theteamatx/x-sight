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

import time
from typing import Optional, Sequence, Tuple

from helpers.logs.logs_handler import logger as logging
from overrides import override
from sight import service_utils as service
from sight.proto import sight_pb2
from sight.widgets.decision.optimizer_client import OptimizerClient
from sight_service.proto import service_pb2
from sight_service.shared_batch_messages import CachedBatchMessages


class LLMOptimizerClient(OptimizerClient):
  """LLM client for the Sight service."""

  def __init__(self, llm_name: str, description: str, sight):
    super().__init__(sight_pb2.DecisionConfigurationStart.OptimizerType.OT_LLM)
    if llm_name.startswith('text_bison'):
      self._algorithm = sight_pb2.DecisionConfigurationStart.LLMConfig.LLMAlgorithm.LA_TEXT_BISON
    elif llm_name.startswith('chat_bison'):
      self._algorithm = sight_pb2.DecisionConfigurationStart.LLMConfig.LLMAlgorithm.LA_CHAT_BISON
    elif llm_name.startswith('gemini_pro'):
      self._algorithm = sight_pb2.DecisionConfigurationStart.LLMConfig.LLMAlgorithm.LA_GEMINI_PRO
    else:
      raise ValueError(f'Unknown LLM Algorithm {llm_name}')

    if llm_name.endswith('_optimize'):
      self._goal = sight_pb2.DecisionConfigurationStart.LLMConfig.LLMGoal.LM_OPTIMIZE
    elif llm_name.endswith('_recommend'):
      self._goal = sight_pb2.DecisionConfigurationStart.LLMConfig.LLMGoal.LM_RECOMMEND
    elif llm_name.endswith('_interactive'):
      self._goal = sight_pb2.DecisionConfigurationStart.LLMConfig.LLMGoal.LM_INTERACTIVE
    else:
      raise ValueError(f'Unknown LLM Goal {llm_name}')

    self._description = description

    self.cache: CachedBatchMessages = CachedBatchMessages()
    self._sight = sight
    self._worker_id = None
    

  @override
  def create_config(self) -> sight_pb2.DecisionConfigurationStart.ChoiceConfig:
    choice_config = sight_pb2.DecisionConfigurationStart.ChoiceConfig()
    llm_config = sight_pb2.DecisionConfigurationStart.LLMConfig(
        algorithm=self._algorithm,
        goal=self._goal,
        description=self._description)
    choice_config.llm_config.CopyFrom(llm_config)
    return choice_config

  @override
  def decision_point(self, sight, request: service_pb2.DecisionPointRequest):
    for key, value in sight.widget_decision_state["state"].items():
      param = request.decision_point.state_params.add()
      param.key = key
      param.value.sub_type
      param.value.double_value = value

    while True:
      response = service.call(
          lambda s, meta: s.DecisionPoint(request, 300, metadata=meta))
      logging.info('decision_point() response=%s' % response)
      if response.action_type == service_pb2.DecisionPointResponse.ActionType.AT_ACT:
        return self._get_dp_action(response)
      if response.action_type == service_pb2.DecisionPointResponse.ActionType.AT_RETRY:
        time.sleep(5)

  @override
  def finalize_episode(self, sight,
                       request: service_pb2.FinalizeEpisodeRequest):
    logging.info('LLMOptimizerClient() finalize_episode, request=%s', request)
    response = service.call(
        lambda s, meta: s.FinalizeEpisode(request, 300, metadata=meta))
    return response
