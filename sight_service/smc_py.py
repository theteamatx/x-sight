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
import queue
import threading
from typing import Any, Dict, List, Tuple

from helpers.logs.logs_handler import logger as logging
import numpy as np
from overrides import overrides
from scipy.stats import uniform
from sight.proto import sight_pb2
from sight_service.optimizer_instance import OptimizerInstance
from sight_service.optimizer_instance import param_dict_to_proto
from sight_service.proto import service_pb2
from smcpy import AdaptiveSampler as Sampler
from smcpy import VectorMCMC
from smcpy import VectorMCMCKernel


# Initialize model
class ModelSamplingDriver():
  '''
  Driver for communicating with SMC.
  '''

  def __init__(self, param_names: List[str], priors: List, std_dev: float):
    self._buf_size = 50
    self._model_inputs_meta_q = queue.Queue(1)
    self._model_inputs_q = queue.Queue(self._buf_size)
    self._model_outputs_meta_q = queue.Queue(1)
    self._model_outputs_q = queue.Queue(self._buf_size)

    # Define prior distributions & MCMC kernel
    self._vector_mcmc = VectorMCMC(self.evaluate, [0], priors, std_dev)
    self._mcmc_kernel = VectorMCMCKernel(self._vector_mcmc,
                                         param_order=param_names)
    self._smc = Sampler(self._mcmc_kernel)
    self._num_mcmc_samples = 5

  def sample(self):
    step_list, mll_list = self._smc.sample(
        num_particles=self._buf_size,
        num_mcmc_samples=self._num_mcmc_samples,
        target_ess=0.8)
    self._model_inputs_meta_q.put(-1)
    # print ('step_list=', step_list.__dict__)
    # print ('step_list=', step_list.mean())
    # print ('mll_list=', mll_list)

    print(f'phi_sequence={self._smc.phi_sequence}')
    print(f'fbf norm index={self._smc.req_phi_index}')
    print('marginal log likelihood = {}'.format(mll_list[-1]))
    print('parameter means = {}'.format(step_list[-1].compute_mean()))

  def evaluate(self, params):
    print('<<< ModelSamplingDriver evaluate() #params=', len(params))
    self._model_inputs_meta_q.put(len(params))
    for i, p in enumerate(params):
      self._model_inputs_q.put({'idx': i, 'params': p})

    results = [None] * len(params)
    for i in range(len(params)):
      v = self._model_outputs_q.get()
      results[v['idx']] = v['result']
    print('>>> ModelSamplingDriver evaluate() #results=', len(results))
    return np.array(results)


class SMCPy(OptimizerInstance):
  """Uses the SMCPy library to choose the parameters of the code.

  Attributes:
    possible_values: Maps each action attributes to the list of possible values
      of this attribute.
  """

  def __init__(self):
    super(SMCPy, self).__init__()
    self.num_samples_issued = 0
    self.active_samples = {}
    self.complete_samples = {}
    self.possible_values = {}
    self._lock = threading.RLock()
    self._driver = None

  @overrides
  def launch(self,
             request: service_pb2.LaunchRequest) -> service_pb2.LaunchResponse:
    response = super(SMCPy, self).launch(request)

    # self.possible_values = {}
    # for i, key in enumerate(sorted(self.actions.keys())):
    #   if self.actions[key].valid_float_values:
    #     self.possible_values[key] = list(self.actions[key].valid_float_values)
    #   elif self.actions[key].step_size:
    #     self.possible_values[key] = []
    #     cur = self.actions[key].min_value
    #     while cur <= self.actions[key].max_value:
    #       self.possible_values[key].append(cur)
    #       cur += self.actions[key].step_size
    # print('possible_values=%s' % self.possible_values)

    self._param_names = list(sorted(self.actions.keys()))
    self._driver = ModelSamplingDriver(param_names=self._param_names,
                                       priors=[
                                           uniform(self.actions[key].min_value,
                                                   self.actions[key].max_value)
                                           for key in self._param_names
                                       ],
                                       std_dev=0.5)
    self._smc_thread = threading.Thread(target=self._driver.sample, args=())
    self._smc_thread.start()

    self._num_samples_in_cur_batch = 0
    self._sample_idx = 0
    self._num_samples_complete = 0
    self._num_samples_remaining = 0

    response.display_string = 'SMCPy Start'
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
    logging.info('DecisionPoint self._lock=%s', self._lock)

    self._lock.acquire()
    logging.info(
        'decision_point() _sample_idx=%s, self._num_samples_in_cur_batch=%s, self._num_samples_remaining=%s, self._num_samples_complete=%s',
        self._sample_idx, self._num_samples_in_cur_batch,
        self._num_samples_remaining, self._num_samples_complete)

    dp_response = service_pb2.DecisionPointResponse()
    logging.info('dp_response=%s', dp_response)
    params = []
    if self._sample_idx == self._num_samples_in_cur_batch and \
        self._num_samples_complete < self._num_samples_remaining:
      logging.info('AT_RETRY')
      self._lock.release()
      dp_response.action_type = service_pb2.DecisionPointResponse.ActionType.AT_RETRY
      return dp_response

    logging.info('Start new batch')
    # Start new batch
    if self._sample_idx == self._num_samples_in_cur_batch:
      logging.info('Starting new batch')
      self._num_samples_in_cur_batch = self._driver._model_inputs_meta_q.get()
      self._sample_idx = 0
      self._num_samples_complete = 0

    logging.info('Getting Params')

    params = self._driver._model_inputs_q.get()['params']

    self.active_samples[request.worker_id] = {
        'action': params,
        'sample_num': self.num_samples_issued,
        'idx': self._sample_idx,
    }
    self._sample_idx += 1

    self.num_samples_issued += 1
    self._lock.release()

    for i, value in enumerate(params):
      a = dp_response.action.add()
      a.key = self._param_names[i]
      a.value.double_value = float(value)

    print('DecisionPoint response=%s' % dp_response)
    dp_response.action_type = service_pb2.DecisionPointResponse.ActionType.AT_ACT
    return dp_response

  @overrides
  def finalize_episode(
      self, request: service_pb2.FinalizeEpisodeRequest
  ) -> service_pb2.FinalizeEpisodeResponse:
    logging.info('FinalizeEpisode request=%s', request)
    d = {}
    for a in request.decision_point.choice_params:
      d[a.key] = a.value.double_value
    result = [d[key] for key in self._param_names]

    self._lock.acquire()
    self._driver._model_outputs_q.put({
        'idx': self.active_samples[request.worker_id]['idx'],
        'result': result,
    })
    self._num_samples_complete += 1

    logging.info('FinalizeEpisode outcome=%s / %s',
                 request.decision_outcome.reward, d)
    del self.active_samples[request.worker_id]
    self._lock.release()
    return service_pb2.FinalizeEpisodeResponse(response_str='Success!')

  @overrides
  def current_status(
      self, request: service_pb2.CurrentStatusRequest
  ) -> service_pb2.CurrentStatusResponse:
    response = '[SMCPy (num_ask=#%s, num_tell=#%s)\n' % (
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

    return service_pb2.CurrentStatusResponse(response_str=response)
