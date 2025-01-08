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

import json
import os
import random
import threading
from typing import Any, Dict, List, Tuple

import google.auth
import google.auth.transport.requests
from helpers.logs.logs_handler import logger as logging
import nevergrad as ng
from overrides import overrides
import requests
from sight.proto import sight_pb2
from sight.utils.proto_conversion import convert_dict_to_proto
from sight_service.normalizer import Normalizer
from sight_service.optimizer_instance import OptimizerInstance
from sight_service.proto import service_pb2

_file_name = "nevergrad_opt.py"


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
    self._total_count = 0
    self._completed_count = 0
    self.normalizer = Normalizer()

  @overrides
  def launch(self,
             request: service_pb2.LaunchRequest) -> service_pb2.LaunchResponse:
    response = super(NeverGradOpt, self).launch(request)
    # logging.info('request : %s', request)
    self._ng_config = request.decision_config_params.choice_config[
        request.label].never_grad_config
    # print ('ng=%s' % ng.__dict__)

    self._total_count = request.decision_config_params.num_trials

    self.actions = self.normalizer.normalize_in_0_to_1(self.actions)
    # print("self.actions : ", self.actions)

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
    # print('possible_values=%s' % self.possible_values)

    params = {}
    for key, p in self.actions.items():
      if self.actions[key].valid_float_values:
        params[key] = ng.p.Choice(choices=len(self.possible_values[key]))
      elif self.actions[key].step_size:
        params[key] = ng.p.TransitionChoice(
            choices=len(self.possible_values[key]))
      else:
        params[key] = ng.p.Scalar(lower=p.min_value, upper=p.max_value)

    # print('here params are  : ', params)
    # # print('here **params are : ', **params)
    # print('here ng.p.Dict is : ', ng.p.Dict(**params))
    # print('here ng.p.Instrumentation is : ', ng.p.Instrumentation(ng.p.Dict(**params)))

    parametrization = ng.p.Instrumentation(ng.p.Dict(**params))
    budget = 1000

    if (self._ng_config.algorithm == sight_pb2.DecisionConfigurationStart.
        NeverGradConfig.NeverGradAlgorithm.NG_AUTO):
      self._optimizer = ng.optimizers.NGOpt(parametrization=parametrization,
                                            budget=budget)
    elif (self._ng_config.algorithm == sight_pb2.DecisionConfigurationStart.
          NeverGradConfig.NeverGradAlgorithm.NG_BO):
      self._optimizer = ng.optimizers.BO(parametrization=parametrization,
                                         budget=budget)
    elif (self._ng_config.algorithm == sight_pb2.DecisionConfigurationStart.
          NeverGradConfig.NeverGradAlgorithm.NG_CMA):
      self._optimizer = ng.optimizers.CMA(parametrization=parametrization,
                                          budget=budget)
    elif (self._ng_config.algorithm == sight_pb2.DecisionConfigurationStart.
          NeverGradConfig.NeverGradAlgorithm.NG_TwoPointsDE):
      self._optimizer = ng.optimizers.TwoPointsDE(
          parametrization=parametrization, budget=budget)
    elif (self._ng_config.algorithm == sight_pb2.DecisionConfigurationStart.
          NeverGradConfig.NeverGradAlgorithm.NG_RandomSearch):
      self._optimizer = ng.optimizers.RandomSearch(
          parametrization=parametrization, budget=budget)
    elif (self._ng_config.algorithm == sight_pb2.DecisionConfigurationStart.
          NeverGradConfig.NeverGradAlgorithm.NG_PSO):
      self._optimizer = ng.optimizers.PSO(parametrization=parametrization,
                                          budget=budget)
    elif (self._ng_config.algorithm == sight_pb2.DecisionConfigurationStart.
          NeverGradConfig.NeverGradAlgorithm.NG_ScrHammersleySearch):
      self._optimizer = ng.optimizers.ScrHammersleySearch(
          parametrization=parametrization, budget=budget)
    elif (self._ng_config.algorithm == sight_pb2.DecisionConfigurationStart.
          NeverGradConfig.NeverGradAlgorithm.NG_DE):
      self._optimizer = ng.optimizers.DE(parametrization=parametrization,
                                         budget=budget)
    elif (self._ng_config.algorithm == sight_pb2.DecisionConfigurationStart.
          NeverGradConfig.NeverGradAlgorithm.NG_CGA):
      self._optimizer = ng.optimizers.cGA(parametrization=parametrization,
                                          budget=budget)
    elif (self._ng_config.algorithm == sight_pb2.DecisionConfigurationStart.
          NeverGradConfig.NeverGradAlgorithm.NG_ES):
      self._optimizer = ng.optimizers.ES(parametrization=parametrization,
                                         budget=budget)
    elif (self._ng_config.algorithm == sight_pb2.DecisionConfigurationStart.
          NeverGradConfig.NeverGradAlgorithm.NG_DL_OPO):
      self._optimizer = ng.optimizers.DiscreteLenglerOnePlusOne(
          parametrization=parametrization, budget=budget)
    elif (self._ng_config.algorithm == sight_pb2.DecisionConfigurationStart.
          NeverGradConfig.NeverGradAlgorithm.NG_DDE):
      self._optimizer = ng.optimizers.DiscreteDE(
          parametrization=parametrization, budget=budget)
    elif (self._ng_config.algorithm == sight_pb2.DecisionConfigurationStart.
          NeverGradConfig.NeverGradAlgorithm.NG_NMM):
      self._optimizer = ng.optimizers.NeuralMetaModel(
          parametrization=parametrization, budget=budget)
    elif (self._ng_config.algorithm == sight_pb2.DecisionConfigurationStart.
          NeverGradConfig.NeverGradAlgorithm.NG_TINY_SPSA):
      self._optimizer = ng.optimizers.TinySPSA(parametrization=parametrization,
                                               budget=budget)
    elif (self._ng_config.algorithm == sight_pb2.DecisionConfigurationStart.
          NeverGradConfig.NeverGradAlgorithm.NG_VORONOI_DE):
      self._optimizer = ng.optimizers.VoronoiDE(parametrization=parametrization,
                                                budget=budget)
    elif (self._ng_config.algorithm == sight_pb2.DecisionConfigurationStart.
          NeverGradConfig.NeverGradAlgorithm.NG_CMA_SMALL):
      self._optimizer = ng.optimizers.CMAsmall(parametrization=parametrization,
                                               budget=budget)

    # print(self._optimizer, type(self._optimizer))

    response.display_string = 'NeverGrad Start'
    print('response=%s' % response)
    print('here nevergrad object : ', self.__dict__)
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
    # logging.info('DecisionPoint request=%s', request)
    # print('DecisionPoint request=%s' % request)

    self._lock.acquire()
    selected_actions = self._optimizer.ask()
    # logging.info('selected_actions=%s', selected_actions.args)

    # logging.info('selected_actions=%s', selected_actions.kwargs)
    self.active_samples[request.worker_id] = {
        'action': selected_actions.args[0],
        'sample_num': self.num_samples_issued,
    }
    # print('self.active_samples : ', self.active_samples)
    self.last_action = selected_actions
    self.num_samples_issued += 1
    self._lock.release()

    denormalized_actions = self.normalizer.denormalize_from_0_to_1(
        selected_actions.args[0])
    # print("denormalized_actions : ", denormalized_actions)

    dp_response = service_pb2.DecisionPointResponse()

    dp_response.action.CopyFrom(
        convert_dict_to_proto(dict=denormalized_actions))

    dp_response.action_type = service_pb2.DecisionPointResponse.ActionType.AT_ACT
    return dp_response

  @overrides
  def finalize_episode(
      self, request: service_pb2.FinalizeEpisodeRequest
  ) -> service_pb2.FinalizeEpisodeResponse:
    # logging.info('FinalizeEpisode request=%s', request)
    d = self.last_action

    self._lock.acquire()
    # logging.info('FinalizeEpisode complete_samples=%s' % self.complete_samples)
    self.complete_samples[self.active_samples[request.worker_id]['sample_num']] = {
        # 'outcome': request.decision_outcome.reward,
        # 'action': self.active_samples[request.worker_id]['action'],
        'reward': request.decision_outcome.reward,
        'action': self.active_samples[request.worker_id]['action'],
        'outcome': request.decision_outcome.outcome_params
    }
    # print('self.complete_samples : ', self.complete_samples)
    del self.active_samples[request.worker_id]

    # logging.info('FinalizeEpisode outcome=%s / %s',
    #              request.decision_outcome.reward, d)
    self._optimizer.tell(d, 0 - request.decision_outcome.reward)
    # self._completed_count += 1

    del self.last_action
    self._lock.release()
    return service_pb2.FinalizeEpisodeResponse(response_str='Success!')

  @overrides
  def current_status(
      self, request: service_pb2.CurrentStatusRequest
  ) -> service_pb2.CurrentStatusResponse:
    response = '[NeverGrad (num_ask=#%s, num_tell=#%s)\n' % (
        self._optimizer.num_ask, self._optimizer.num_tell)

    self._lock.acquire()
    response += 'sample_num, ' + ', '.join(list(self.actions)) + ', outcome\n'
    cur = [0] * len(self.actions)
    keys = sorted(self.actions.keys())
    logging.info('self.complete_samples=%s', self.complete_samples)
    for s in sorted(self.complete_samples.items(),
                    key=lambda x: x[1]['outcome'],
                    reverse=True):
      response += str(s[0]) + ', '
      response += ', '.join([str(s[1]['action'][key]) for key in keys])
      response += ', ' + str(s[1]['outcome']) + '\n'

    response += 'pareto_front:\n'
    for trial in self._optimizer.pareto_front():
      response += ', '.join([str(trial.args[0][key]) for key in keys]) + '\n'
    response += ']\n'
    self._lock.release()

    # print('self._total_count was : ', self._total_count)
    # print('self._completed_count is now : ', self._completed_count)

    if (self._completed_count == self._total_count):
      status = service_pb2.CurrentStatusResponse.Status.SUCCESS
    elif (self._completed_count < self._total_count):
      status = service_pb2.CurrentStatusResponse.Status.IN_PROGRESS
    else:
      status = service_pb2.CurrentStatusResponse.Status.FAILURE

    return service_pb2.CurrentStatusResponse(response_str=response,
                                             status=status)

  @overrides
  def WorkerAlive(
      self, request: service_pb2.WorkerAliveRequest
  ) -> service_pb2.WorkerAliveResponse:
    method_name = "WorkerAlive"
    logging.debug(">>>>  In %s of %s", method_name, _file_name)
    if (self._completed_count == self._total_count):
      worker_alive_status = service_pb2.WorkerAliveResponse.StatusType.ST_DONE
    # elif(not self.pending_samples):
    #    worker_alive_status = service_pb2.WorkerAliveResponse.StatusType.ST_RETRY
    else:
      # Increasing count here so that multiple workers can't enter the dp call for same sample at last
      self._completed_count += 1
      worker_alive_status = service_pb2.WorkerAliveResponse.StatusType.ST_ACT
    logging.info("worker_alive_status is %s", worker_alive_status)
    logging.debug("<<<<  Out %s of %s", method_name, _file_name)
    return service_pb2.WorkerAliveResponse(status_type=worker_alive_status)
