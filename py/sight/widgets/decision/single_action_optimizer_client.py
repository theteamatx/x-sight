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
"""Client for optimizers that are called once per episode to communicate with server."""
import time
from typing import Optional, Sequence, Tuple

from helpers.logs.logs_handler import logger as logging
from overrides import override
from sight import service_utils as service
from sight.proto import sight_pb2
from sight.utils.proto_conversion import convert_proto_to_dict
from sight.utils.proto_conversion import update_proto_map
from sight.widgets.decision.optimizer_client import OptimizerClient
from sight_service.proto import service_pb2


class SingleActionOptimizerClient(OptimizerClient):
  """Single-action Client for the Sight service."""

  def __init__(
      self,
      optimizer_type: sight_pb2.DecisionConfigurationStart.OptimizerType,
      sight,
      algorithm=None):
    super().__init__(optimizer_type)
    self._sight = sight
    self._last_action = None
    self.exp_completed = False
    if (algorithm == None):
      self._algorithm = algorithm
    elif algorithm == 'auto':
      self._algorithm = sight_pb2.DecisionConfigurationStart.NeverGradConfig.NeverGradAlgorithm.NG_AUTO
    elif algorithm == 'bo':
      self._algorithm = sight_pb2.DecisionConfigurationStart.NeverGradConfig.NeverGradAlgorithm.NG_BO
    elif algorithm == 'cma':
      self._algorithm = sight_pb2.DecisionConfigurationStart.NeverGradConfig.NeverGradAlgorithm.NG_CMA
    elif algorithm == 'two_points_de':
      self._algorithm = sight_pb2.DecisionConfigurationStart.NeverGradConfig.NeverGradAlgorithm.NG_TwoPointsDE
    elif algorithm == 'random_search':
      self._algorithm = sight_pb2.DecisionConfigurationStart.NeverGradConfig.NeverGradAlgorithm.NG_RandomSearch
    elif algorithm == 'pso':
      self._algorithm = sight_pb2.DecisionConfigurationStart.NeverGradConfig.NeverGradAlgorithm.NG_PSO
    elif algorithm == 'scr_hammersley_search':
      self._algorithm = sight_pb2.DecisionConfigurationStart.NeverGradConfig.NeverGradAlgorithm.NG_ScrHammersleySearch
    elif algorithm == 'de':
      self._algorithm = sight_pb2.DecisionConfigurationStart.NeverGradConfig.NeverGradAlgorithm.NG_DE
    elif algorithm == 'cga':
      self._algorithm = sight_pb2.DecisionConfigurationStart.NeverGradConfig.NeverGradAlgorithm.NG_CGA
    elif algorithm == 'es':
      self._algorithm = sight_pb2.DecisionConfigurationStart.NeverGradConfig.NeverGradAlgorithm.NG_ES
    elif algorithm == 'dl_opo':
      self._algorithm = sight_pb2.DecisionConfigurationStart.NeverGradConfig.NeverGradAlgorithm.NG_DL_OPO
    elif algorithm == 'dde':
      self._algorithm = sight_pb2.DecisionConfigurationStart.NeverGradConfig.NeverGradAlgorithm.NG_DDE
    elif algorithm == 'nmm':
      self._algorithm = sight_pb2.DecisionConfigurationStart.NeverGradConfig.NeverGradAlgorithm.NG_NMM
    elif algorithm == 'tiny_spsa':
      self._algorithm = sight_pb2.DecisionConfigurationStart.NeverGradConfig.NeverGradAlgorithm.NG_TINY_SPSA
    elif algorithm == 'voronoi_de':
      self._algorithm = sight_pb2.DecisionConfigurationStart.NeverGradConfig.NeverGradAlgorithm.NG_VORONOI_DE
    elif algorithm == 'cma_small':
      self._algorithm = sight_pb2.DecisionConfigurationStart.NeverGradConfig.NeverGradAlgorithm.NG_CMA_SMALL
    else:
      raise ValueError(f'Unsupported NeverGrad Algorithm {algorithm}')

  @override
  def create_config(self) -> sight_pb2.DecisionConfigurationStart.ChoiceConfig:
    choice_config = sight_pb2.DecisionConfigurationStart.ChoiceConfig()
    if (self._algorithm):
      ng_config = sight_pb2.DecisionConfigurationStart.NeverGradConfig(
          algorithm=self._algorithm)
      choice_config.never_grad_config.CopyFrom(ng_config)
    return choice_config

  @override
  def decision_point(self, sight, request: service_pb2.DecisionPointRequest):
    # while True:
    response = service.call(
        lambda s, meta: s.DecisionPoint(request, 300, metadata=meta))
    if response.action_type == service_pb2.DecisionPointResponse.ActionType.AT_ACT:
      self._last_action = response.action
      return self._get_dp_action(response)
    # elif response.action_type == service_pb2.DecisionPointResponse.ActionType.AT_DONE:
    #   self.exp_completed = True
    #   return None
    # elif response.action_type == service_pb2.DecisionPointResponse.ActionType.AT_RETRY:
    #   print('waiting in decision point to get server from response......')
    #   logging.info('sleeping for 5 seconds......')
    #   time.sleep(5)
    else:
      raise ValueError("No action received from server")

  @override
  def finalize_episode(self, sight,
                       request: service_pb2.FinalizeEpisodeRequest):
    logging.info('SingleActionOptimizerClient() finalize_episode')
    response = service.call(
        lambda s, meta: s.FinalizeEpisode(request, 300, metadata=meta))
    return response
