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

import nevergrad as ng
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


class NeverGradOpt(OptimizerInstance):
  """Uses the NeverGrad library to choose the parameters of the code.

  Attributes:
    possible_values: Maps each action attributes to the list of possible values
      of this attribute.
  """

  def __init__(self):
    super().__init__()
    self.num_samples_issued = 0
    self.active_samples = {}
    self.complete_samples = {}
    self.possible_values = {}
    self._lock = threading.RLock()

  @overrides
  def launch(
      self, request: service_pb2.LaunchRequest
  ) -> service_pb2.LaunchResponse:
    response = super(NeverGradOpt, self).launch(request)
    print ('ng=%s' % ng.__dict__)
    
    self.possible_values = {}
    for i, key in enumerate(sorted(self.actions.keys())):
      if self.actions[key].valid_float_values:
        self.possible_values[key] = list(self.actions[key].valid_float_values)
      elif self.actions[key].step_size:
        self.possible_values[key] = []
        cur = self.actions[key].min_value
        while cur <= self.actions[key].max_value:
          self.possible_values[key].append(cur)
          cur += self.actions[key].step_size
    print('possible_values=%s' % self.possible_values)
    
    params = {}
    for key, p in self.actions.items():
      if self.actions[key].valid_float_values:
        params[key] = ng.p.Choice(choices=len(self.possible_values[key]))
      elif self.actions[key].step_size:
        params[key] = ng.p.TransitionChoice(choices=len(self.possible_values[key]))
      else:
        params[key] = ng.p.Scalar(lower=p.min_value, upper=p.max_value)

    self._optimizer = ng.optimizers.NGOpt(
        parametrization=ng.p.Instrumentation(ng.p.Dict(**params)), 
        budget=100)
    response.display_string = 'NeverGrad Start'
    print('response=%s' % response)
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
    selected_actions = self._optimizer.ask()
    logging.info('selected_actions=%s', selected_actions.args)
    logging.info('selected_actions=%s', selected_actions.kwargs)
    self.active_samples[request.worker_id] = {
        'action': selected_actions.args[0],
        'sample_num': self.num_samples_issued,
    }
    self.num_samples_issued += 1
    self._lock.release()

    dp_response = service_pb2.DecisionPointResponse()
    for key, value in selected_actions.args[0].items():
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
    # logging.info('FinalizeEpisode complete_samples=%s' % self.complete_samples)
    self.complete_samples[self.active_samples[request.worker_id]['sample_num']] = {
        'outcome': request.decision_outcome.outcome_value,
        'action': self.active_samples[request.worker_id]['action'],
    }
    del self.active_samples[request.worker_id]
    
    logging.info('FinalizeEpisode outcome=%s / %s', request.decision_outcome.outcome_value, d)
    self._optimizer.tell(d, 0-request.decision_outcome.outcome_value)
    self._lock.release()
    return service_pb2.FinalizeEpisodeResponse(response_str='Success!')

  @overrides
  def current_status(
      self, request: service_pb2.CurrentStatusRequest
  ) -> service_pb2.CurrentStatusResponse:
    response = '[NeverGrad (num_ask=#%s, num_tell=#%s)\n' % (self._optimizer.num_ask, self._optimizer.num_tell)
    
    self._lock.acquire()
    response += 'sample_num, ' + ', '.join(list(self.actions)) + ', outcome\n'
    cur = [0] * len(self.actions)
    keys = sorted(self.actions.keys())
    logging.info('self.complete_samples=%s', self.complete_samples)
    for s in sorted(self.complete_samples.items(), key=lambda x: x[1]['outcome'], reverse=True):
      response += str(s[0])+', '
      response += ', '.join([str(s[1]['action'][key]) for key in keys])
      response += ', '+str(s[1]['outcome'])+'\n'
    
    response += 'pareto_front:\n'
    for trial in self._optimizer.pareto_front():
      response += ', '.join([str(trial.args[0][key]) for key in keys])+'\n'
    response += ']\n'
    self._lock.release()
    
    return service_pb2.CurrentStatusResponse(response_str=response)