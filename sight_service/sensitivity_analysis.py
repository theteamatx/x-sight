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
"""Sensitivity analysis of Sight applications."""

from helpers.logs.logs_handler import logger as logging
import random
from typing import Any, Dict, List, Tuple
from overrides import overrides
from sight_service.optimizer_instance import OptimizerInstance
from sight_service.optimizer_instance import param_dict_to_proto
from sight_service.proto import service_pb2
import threading

_file_name = 'sensitivity_analysis.py'


class SensitivityAnalysis(OptimizerInstance):
    """Exhaustively searches over all the possible values of the action attributes.

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
            self,
            request: service_pb2.LaunchRequest) -> service_pb2.LaunchResponse:
        method_name = 'launch'
        logging.debug('>>>>  In %s of %s', method_name, _file_name)

        response = super(SensitivityAnalysis, self).launch(request)
        self.possible_values = {}
        for i, key in enumerate(sorted(self.actions.keys())):
            if self.actions[key].valid_float_values:
                self.possible_values[key] = list(
                    self.actions[key].valid_float_values)
            elif self.actions[key].step_size:
                self.possible_values[key] = []
                cur = self.actions[key].min_value
                while cur <= self.actions[key].max_value:
                    self.possible_values[key].append(cur)
                    cur += self.actions[key].step_size

        logging.info('possible_values=%s', self.possible_values)
        response.display_string = 'Sensitivity Analysis!'
        logging.debug('<<<<  Out %s of %s', method_name, _file_name)
        return response

    @overrides
    def decision_point(
        self, request: service_pb2.DecisionPointRequest
    ) -> service_pb2.DecisionPointResponse:
        method_name = 'decision_point'
        logging.debug('>>>>  In %s of %s', method_name, _file_name)

        next_action = {}
        for i, key in enumerate(self.actions):
            if key in self.possible_values:
                print('selecting from possible values')
                next_action[key] = self.possible_values[key][random.randint(
                    0,
                    len(self.possible_values[key]) - 1)]
            elif self.actions[key].HasField('continuous_prob_dist'):
                rand_val = random.gauss(
                    self.actions[key].continuous_prob_dist.gaussian.mean,
                    self.actions[key].continuous_prob_dist.gaussian.stdev)
                print(
                    'self.actions[key].continuous_prob_dist=%s, rand_val=%s' %
                    (self.actions[key].continuous_prob_dist, rand_val))
                if rand_val < self.actions[key].min_value:
                    rand_val = self.actions[key].min_value
                elif rand_val > self.actions[key].max_value:
                    rand_val = self.actions[key].max_value
                next_action[key] = rand_val
                if self.actions[key].continuous_prob_dist.HasField('gaussian'):
                    rand_val = random.gauss(
                        self.actions[key].continuous_prob_dist.gaussian.mean,
                        self.actions[key].continuous_prob_dist.gaussian.stdev)
                    print(
                        'self.actions[key].continuous_prob_dist=%s, rand_val=%s'
                        % (self.actions[key].continuous_prob_dist, rand_val))
                    if rand_val < self.actions[key].min_value:
                        rand_val = self.actions[key].min_value
                    elif rand_val > self.actions[key].max_value:
                        rand_val = self.actions[key].max_value
                    next_action[key] = rand_val
                elif self.actions[key].continuous_prob_dist.HasField(
                        'uniform'):
                    rand_val = random.uniform(
                        self.actions[key].continuous_prob_dist.uniform.min_val,
                        self.actions[key].continuous_prob_dist.uniform.max_val)
                    print(
                        'self.actions[key].continuous_prob_dist=%s, rand_val=%s'
                        % (self.actions[key].continuous_prob_dist, rand_val))
                    next_action[key] = rand_val
                else:
                    raise ValueError(
                        'Only support Gaussian continuous distribution.')
            elif self.actions[key].HasField('discrete_prob_dist'):
                if self.actions[key].discrete_prob_dist.HasField('uniform'):
                    rand_val = random.randint(
                        self.actions[key].discrete_prob_dist.uniform.min_val,
                        self.actions[key].discrete_prob_dist.uniform.max_val)
                    print(
                        'self.actions[key].discrete_prob_dist=%s, rand_val=%s'
                        % (self.actions[key].discrete_prob_dist, rand_val))
                    next_action[key] = rand_val
                else:
                    raise ValueError(
                        'Only support Uniform discrete distribution.')
            else:
                print('selecting from random.uniform')
                next_action[key] = random.uniform(self.actions[key].min_value,
                                                  self.actions[key].max_value)

        self._lock.acquire()
        self.active_samples[request.worker_id] = {
            'action': next_action,
            'sample_num': self.num_samples_issued,
        }
        self.num_samples_issued += 1
        self._lock.release()

        logging.info('next_action=%s', next_action)
        dp_response = service_pb2.DecisionPointdp_response()
        dp_response.action.extend(param_dict_to_proto(next_action))
        dp_response.action_type = service_pb2.DecisionPointResponse.ActionType.AT_ACT
        logging.debug('<<<<  Out %s of %s', method_name, _file_name)
        return dp_response

    @overrides
    def finalize_episode(
        self, request: service_pb2.FinalizeEpisodeRequest
    ) -> service_pb2.FinalizeEpisodeResponse:
        method_name = 'finalize_episode'
        logging.debug('>>>>  In %s of %s', method_name, _file_name)
        # logging.info('Running for exhaustive search....')

        self._lock.acquire()
        # logging.info('FinalizeEpisode complete_samples=%s' % self.complete_samples)
        self.complete_samples[self.active_samples[
            request.worker_id]['sample_num']] = {
                'outcome': request.decision_outcome.outcome_value,
                'action': self.active_samples[request.worker_id]['action'],
            }
        del self.active_samples[request.worker_id]
        self._lock.release()

        # logging.info('FinalizeEpisode active_samples=%s' % self.active_samples)
        logging.debug('<<<<  Out %s of %s', method_name, _file_name)
        return service_pb2.FinalizeEpisodeResponse(response_str='Success!')

    @overrides
    def current_status(
        self, request: service_pb2.CurrentStatusRequest
    ) -> service_pb2.CurrentStatusResponse:
        method_name = 'current_status'
        logging.debug('>>>>  In %s of %s', method_name, _file_name)
        response = ('[SensitivityAnalysis:\n')
        response += f'  #active_samples={len(self.active_samples)}\n'
        response += '  completed_samples=\n'
        response += 'sample_num, ' + ', '.join(list(
            self.actions)) + ', outcome\n'

        cur = [0] * len(self.actions)
        keys = sorted(self.actions.keys())
        logging.info('self.complete_samples=%s', self.complete_samples)
        for s in sorted(self.complete_samples.items(),
                        key=lambda x: x[1]['outcome'],
                        reverse=True):
            response += str(s[0]) + ', '
            response += ', '.join([str(s[1]['action'][key]) for key in keys])
            response += ', ' + str(s[1]['outcome']) + '\n'
        response += ']'
        logging.debug('<<<<  Out %s of %s', method_name, _file_name)
        return service_pb2.CurrentStatusResponse(response_str=response)

    @overrides
    def fetch_optimal_action(
        self, request: service_pb2.FetchOptimalActionRequest
    ) -> service_pb2.FetchOptimalActionResponse:
        method_name = 'fetch_optimal_action'
        return service_pb2.CurrentStatusResponse(response_str='')
