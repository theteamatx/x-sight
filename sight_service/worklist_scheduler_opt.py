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
"""Exhaustive search for driving Sight applications."""

import logging
from overrides import overrides
from typing import Any, Dict, List, Tuple

from sight.proto import sight_pb2
from sight_service.proto import service_pb2
from sight_service.single_action_optimizer import SingleActionOptimizer
from sight_service.optimizer_instance import param_dict_to_proto
# from sight_service.optimizer_instance import OptimizerInstance
from sight_service.optimizer_instance import param_proto_to_dict
import threading
from sight.widgets.decision import utils


_file_name = "exhaustive_search.py"


class WorklistScheduler(SingleActionOptimizer):
    """Exhaustively searches over all the possible values of the action attributes.

  Attributes:
    possible_values: Maps each action attributes to the list of possible values
      of this attribute.
  """

    def __init__(self):
        super().__init__()
        self.next_sample_to_issue = []
        # self.action_ids = []
        # self.unique_id = 1
        # self.pending_samples = {}
        # self.completed_samples = {}
        # self.active_samples = {}
        self.last_sample = False
        self.sweep_issue_done = False
        self.possible_values = {}
        self.max_reward_sample = {}
        self._lock = threading.RLock()

    @overrides
    def launch(
            self,
            request: service_pb2.LaunchRequest) -> service_pb2.LaunchResponse:
        method_name = "launch"
        logging.debug(">>>>  In %s of %s", method_name, _file_name)

        # print("request in launch is : ", request)
        response = super(WorklistScheduler, self).launch(request)
        # print("self.actions : ", self.actions)
        # self.next_sample_to_issue = [0] * len(self.actions)
        # print("self.next_sample_to_issue : ", self.next_sample_to_issue)

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
        # logging.info('possible_values=%s', self.possible_values)

        # print('self.actions : ', self.actions)
        # print('self.state : ', self.state)
        response.display_string = 'Worklist Scheduler SUCCESS!'
        logging.debug("<<<<  Out %s of %s", method_name, _file_name)
        return response

    @overrides
    def propose_action(
        self, request: service_pb2.ProposeActionRequest
    ) -> service_pb2.ProposeActionResponse:
        # print('request in propose actions: ', request)

        attributes = param_proto_to_dict(request.attributes)
        action_attrs = param_proto_to_dict(request.action_attrs)

        #TODO : working for unique_ids
        # for action in request.actions:
            # print('action.action_attrs : ', action.action_attrs)
        self._lock.acquire()
        self.pending_samples[self.unique_id] = [
            action_attrs, attributes
        ]

        print('self.pending_samples : ',
              self.pending_samples)
        print('self.active_samples : ',
              self.active_samples)
        print('self.completed_samples : ',
              self.completed_samples)
        print('self.unique_id : ', self.unique_id)


        # Create response
        response = service_pb2.ProposeActionResponse(
            action_id=self.unique_id)
        self.unique_id += 1
        self._lock.release()
        return response

    @overrides
    def GetOutcome(
        self, request: service_pb2.GetOutcomeRequest
    ) -> service_pb2.GetOutcomeResponse:
        print('self.pending_samples : ',
              self.pending_samples)
        print('self.active_samples : ',
              self.active_samples)
        print('self.completed_samples : ',
              self.completed_samples)

        response = service_pb2.GetOutcomeResponse()
        if (request.unique_ids):
            required_samples = list(request.unique_ids)
            for sample_id in required_samples:
                outcome = response.outcome.add()
                outcome.action_id = sample_id
                if (sample_id in self.completed_samples):
                    sample_details = self.completed_samples[sample_id]
                    outcome.status = service_pb2.GetOutcomeResponse.Outcome.Status.COMPLETED
                    outcome.reward = sample_details['reward']
                    outcome.action_attrs.extend(param_dict_to_proto(
                        sample_details['action']))
                    outcome.outcome_attrs.extend(param_dict_to_proto(
                        sample_details['outcome']))
                    outcome.attributes.extend(param_dict_to_proto(
                        sample_details['attribute']))
                elif (sample_id in self.pending_samples):
                    outcome.status = service_pb2.GetOutcomeResponse.Outcome.Status.PENDING
                    outcome.response_str = '!! requested sample not yet assigned to any worker !!'
                elif (sample_id in self.active_samples.values()):
                    outcome.status = service_pb2.GetOutcomeResponse.Outcome.Status.ACTIVE
                    outcome.response_str = '!! requested sample not completed yet !!'
                else:
                    outcome.status = service_pb2.GetOutcomeResponse.Outcome.Status.NOT_EXIST
                    outcome.response_str = f'!! requested sample Id {sample_id} does not exist !!'
                    print("!! NOT EXIST !!")
        else:
            for sample_id in self.completed_samples.keys():
                sample_details = self.completed_samples[sample_id]
                outcome = response.outcome.add()
                outcome.action_id = sample_id
                outcome.status = service_pb2.GetOutcomeResponse.Outcome.Status.COMPLETED
                outcome.reward = sample_details['reward']

                outcome.action_attrs.extend(param_dict_to_proto(
                        sample_details['action']))

                outcome.outcome_attrs.extend(param_dict_to_proto(
                        sample_details['outcome']))

                outcome.attributes.extend(param_dict_to_proto(
                        sample_details['attribute']))

        # print('response here: ', response)
        return response

    @overrides
    def decision_point(
        self, request: service_pb2.DecisionPointRequest
    ) -> service_pb2.DecisionPointResponse:
        method_name = "decision_point"
        logging.debug(">>>>  In %s of %s", method_name, _file_name)

        print('self.pending_samples : ',
              self.pending_samples)
        print('self.active_samples : ',
              self.active_samples)
        print('self.completed_samples : ',
              self.completed_samples)
        print('self.unique_id : ', self.unique_id)

        dp_response = service_pb2.DecisionPointResponse()
        if self.pending_samples:
          self._lock.acquire()
          # todo : meetashah : add logic to fetch action stored from propose actions and send it as repsonse
          # key, sample = self.pending_samples.popitem()
          # fetching the key in FIFO manner
          key = next(iter(self.pending_samples))
          sample = self.pending_samples.pop(key)

          self.active_samples[request.worker_id] = {'id': key, 'sample': sample}
          self._lock.release()

          next_action = sample[0]
          logging.info('next_action=%s', next_action)
          # raise SystemExit
          dp_response.action.extend(param_dict_to_proto(next_action))
          print('self.active_samples : ', self.active_samples)
          print('self.pending_samples : ', self.pending_samples)
          print('self.completed_samples : ', self.completed_samples)
          dp_response.action_type = service_pb2.DecisionPointResponse.ActionType.AT_ACT
        else:
          dp_response.action_type = service_pb2.DecisionPointResponse.ActionType.AT_RETRY

        logging.debug("<<<<  Out %s of %s", method_name, _file_name)
        return dp_response

    @overrides
    def finalize_episode(
        self, request: service_pb2.FinalizeEpisodeRequest
    ) -> service_pb2.FinalizeEpisodeResponse:
        method_name = "finalize_episode"
        logging.debug(">>>>  In %s of %s", method_name, _file_name)

        # logging.info("req in finalize episode of dummy.py : %s", request)
        self._lock.acquire()
        sample_dict = self.active_samples[request.worker_id]
        self.completed_samples[sample_dict['id']] = {
            # 'action': self.pending_samples[unique_action_id],
            'action':
            param_proto_to_dict(request.decision_point.choice_params),
            'attribute': sample_dict['sample'][1],
            'reward': request.decision_outcome.reward,
            'outcome':
            param_proto_to_dict(request.decision_outcome.outcome_params)
        }
        del self.active_samples[request.worker_id]
        # self.completed_samples[
        #     unique_action_id-1
        # ] = {
        #     'reward': request.decision_outcome.reward+10,
        #     # 'action': self.pending_samples[unique_action_id],
        #     'action': param_proto_to_dict(request.decision_point.choice_params),
        #     'outcome': param_proto_to_dict(request.decision_outcome.outcome_params)
        # }
        logging.info('FinalizeEpisode completed_samples=%s' %
                     self.completed_samples)
        self._lock.release()

        print('self.active_samples : ', self.active_samples)
        print('self.pending_samples : ', self.pending_samples)
        print('self.completed_samples : ', self.completed_samples)
        logging.debug("<<<<  Out %s of %s", method_name, _file_name)
        return service_pb2.FinalizeEpisodeResponse(response_str='Success!')

    @overrides
    def current_status(
        self, request: service_pb2.CurrentStatusRequest
    ) -> service_pb2.CurrentStatusResponse:
        method_name = "current_status"
        logging.debug(">>>>  In %s of %s", method_name, _file_name)
        response = (
            '[ExhaustiveSearch: {"Done" if self.sweep_issue_done else "In'
            ' Progress"}\n')
        self._lock.acquire()
        response += f'  #pending_samples={len(self.pending_samples)}\n'
        response += '  completed_samples=\n'
        response += ', '.join(list(self.actions)) + ', outcome\n'

        cur = [0] * len(self.actions)
        # action_keys = list(self.actions.keys())
        keys = sorted(self.actions.keys())
        logging.info('self.completed_samples=%s', self.completed_samples)

        reached_last = False
        while not reached_last:
            logging.info('cur(#%d)=%s', len(cur), cur)
            response += ', '.join([
                str(self.possible_values[key][cur[i]])
                for i, key in enumerate(keys)
            ])
            if tuple(cur) in self.completed_samples:
                response += ', ' + str(
                    self.completed_samples[tuple(cur)]['outcome'])
            else:
                response += ', ?'
            response += '\n'

            # Advance cur, starting from the last dimension and going to the first.
            for i, key in reversed(list(enumerate(keys))):
                logging.info(
                    'i=%d, key=%s, cur=%s, self.possible_values[key]=%s',
                    i,
                    key,
                    cur[i],
                    self.possible_values[key],
                )
                if cur[i] < len(self.possible_values[key]) - 1:
                    cur[i] += 1
                    break
                else:
                    cur[i] = 0
                    if i == 0:
                        reached_last = True
        self._lock.release()

        response += ']'
        logging.debug("<<<<  Out %s of %s", method_name, _file_name)
        return service_pb2.CurrentStatusResponse(response_str=response)

    @overrides
    def fetch_optimal_action(
        self, request: service_pb2.FetchOptimalActionRequest
    ) -> service_pb2.FetchOptimalActionResponse:
        method_name = "fetch_optimal_action"
        logging.debug(">>>>  In %s of %s", method_name, _file_name)
        best_action = self.max_reward_sample
        print(" : ", best_action)
        logging.debug("<<<<  Out %s of %s", method_name, _file_name)
        return service_pb2.CurrentStatusResponse(response_str=str(best_action))
