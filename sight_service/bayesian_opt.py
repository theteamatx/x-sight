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

"""LLM-based optimization for driving Sight applications."""

import logging
from overrides import overrides
from typing import Any, Dict, List, Tuple

from bayes_opt import BayesianOptimization
from bayes_opt import UtilityFunction
from sight_service.optimizer_instance import param_dict_to_proto
from sight_service.optimizer_instance import OptimizerInstance
from sight_service.proto import service_pb2
from sight.proto import sight_pb2
import random
import requests
import google.auth
import google.auth.transport.requests
import json
import os
import threading


class BayesianOpt(OptimizerInstance):
  """Uses an LLM to choose the parameters of the code.
  """

  def __init__(self):
    super().__init__()
    self._lock = threading.RLock()

  @overrides
  def launch(
      self, request: service_pb2.LaunchRequest
  ) -> service_pb2.LaunchResponse:
    response = super(BayesianOpt, self).launch(request)
    self._optimizer = BayesianOptimization(
      f=None,
      pbounds={key: (p.min_value, p.max_value) for key, p in self.actions.items()},
      verbose=2,
      allow_duplicate_points=True,
      # random_state=1,
    )
    self._utility = UtilityFunction(kind='ucb', kappa=1.96, xi=0.01)
    response.display_string = 'BayesianOpt Start'
    return response

  def _params_to_dict(self, dp: sight_pb2) -> Dict[str, float]:
    """Returns the dict representation of a DecisionParams proto"""
    d = {}
    for a in dp:
      d[a.key] = a.value.double_value
    return d

  @overrides
  def decision_point(
      self, request: service_pb2.DecisionPointRequest
  ) -> service_pb2.DecisionPointResponse:
    logging.info('DecisionPoint request=%s', request)
    print('DecisionPoint request=%s' % request)

    self._lock.acquire()
    selected_actions = self._optimizer.suggest(self._utility)
    self._lock.release()

    dp_response = service_pb2.DecisionPointResponse()
    for key, value in selected_actions.items():
      a = dp_response.action.add()
      a.key = key
      a.value.double_value = float(value)

    # self.last_outcome = request.decision_outcome.outcome_value
    print('DecisionPoint response=%s' % dp_response)
    return dp_response

  @overrides
  def finalize_episode(
      self, request: service_pb2.FinalizeEpisodeRequest
  ) -> service_pb2.FinalizeEpisodeResponse:
    logging.info('FinalizeEpisode request=%s', request)
    # self._append_outcome(request.decision_outcome.outcome_value)
    # self.history[-1]['outcome'] = request.decision_outcome.outcome_value
    # self.last_outcome = request.decision_outcome.outcome_value
    d = {}
    for a in request.decision_point.choice_params:
      d[a.key] = a.value.double_value

    self._lock.acquire()
    logging.info('FinalizeEpisode outcome=%s / %s', request.decision_outcome.outcome_value, d)
    self._optimizer.register(
        params=d,
        target=request.decision_outcome.outcome_value)
    self._lock.release()
    return service_pb2.FinalizeEpisodeResponse(response_str='Success!')

  @overrides
  def current_status(
      self, request: service_pb2.CurrentStatusRequest
  ) -> service_pb2.CurrentStatusResponse:
    output = '[BayesianOpt (#%s trials)\n' % len(self._optimizer.res)
    for trial in sorted(self._optimizer.res, key=lambda x: x['target'], reverse=True):
      output += '   '+str(trial) + '\n'
    output += ']\n'
    return service_pb2.CurrentStatusResponse(response_str=output)
