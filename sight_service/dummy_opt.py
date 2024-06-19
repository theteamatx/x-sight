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

from sight_service.proto import service_pb2
from sight_service.optimizer_instance import param_dict_to_proto
from sight_service.optimizer_instance import OptimizerInstance
from sight_service.optimizer_instance import param_proto_to_dict
import threading

_file_name = "exhaustive_search.py"

class Dummy(OptimizerInstance):
  """Exhaustively searches over all the possible values of the action attributes.

  Attributes:
    possible_values: Maps each action attributes to the list of possible values
      of this attribute.
  """

  def __init__(self):
    super().__init__()
    self.next_sample_to_issue = []
    self.action_ids = []
    self.unique_id = 1
    self.active_samples = {}
    self.complete_samples = {}
    self.worker_action_id_mapping = {}
    self.last_sample = False
    self.sweep_issue_done = False
    self.possible_values = {}
    self.max_reward_sample = {}
    self._lock = threading.RLock()

  @overrides
  def launch(
      self, request: service_pb2.LaunchRequest
  ) -> service_pb2.LaunchResponse:
    method_name = "launch"
    logging.debug(">>>>  In %s of %s", method_name, _file_name)

    # print("request in launch is : ", request)
    response = super(Dummy, self).launch(request)
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

    response.display_string = 'Dummy SUCCESS!'
    logging.debug("<<<<  Out %s of %s", method_name, _file_name)
    return response

  @overrides
  def propose_action(
    self, request: service_pb2.ProposeActionRequest
  ) -> service_pb2.ProposeActionResponse:
    # unique_action_dict, unique_ids = self.process_actions(request)
    # print('self.action_ids : ', self.action_ids)
    # print('request in propose actions: ', request)
    for action in request.actions:
        # print('action.action_attrs : ', action.action_attrs)
        self.active_samples[self.unique_id] = param_proto_to_dict(action.action_attrs)
        self.action_ids.append(self.unique_id)
        self.unique_id += 1

    print('self.active_samples : ', self.active_samples)  # For demonstration purposes
    print('self.action_ids : ', self.action_ids)  # For demonstration purposes

    # Create response
    response = service_pb2.ProposeActionResponse(unique_ids=self.action_ids)
    return response

  @overrides
  def GetOutcome(self, request: service_pb2.GetOutcomeRequest) -> service_pb2.GetOutcomeResponse:
    # print('request  : ', request)
    # raise SystemExit
    # print('self.completed_actions : ', self.complete_samples)
    # print('self.state : ', self.state)
    # print('self.outcomes : ', self.outcomes)

    response = service_pb2.GetOutcomeResponse()
    if(request.unique_ids):
      required_samples = list(request.unique_ids)
      # print('required_samples : ', required_samples)
      for sample_id in required_samples:
        # print('sample_id : ', sample_id)
        if(sample_id in self.complete_samples):
          sample_details = self.complete_samples[sample_id]
          # print('sample_details : ', sample_details)
          outcome = response.outcome.add()
          outcome.reward = sample_details['reward']
          # for k,v in sample_details['action'].items():
          #   outcome.action_attrs[k] = v
          # for k,v in sample_details['outcome'].items():
          #   outcome.outcome_attrs[k] = v

          # for k,v in sample_details['action'].items():
          outcome.action_attrs = param_dict_to_proto(sample_details['action'])
          # for k,v in sample_details['outcome'].items():
          outcome.outcome_attrs = param_dict_to_proto(sample_details['outcome'])
        elif(sample_id in self.active_samples):
          response.response_str = '!! requested sample not yet assigned to any worker !!'
        elif(sample_id in self.worker_action_id_mapping.values()):
          response.response_str = '!! requested sample not yet completed !!'
        else:
          response.response_str = '!! requested sample Id does not exist !!'


          # print("!!!!!! requested sample not yet completed !!!!!")

        # outcome = response.outcome.add()

        # # # Populate state_attrs map
        # outcome.state_attrs['state_attr1'] = 0.1 * (i + 1)
        # outcome.state_attrs['state_attr2'] = 0.2 * (i + 1)

        # # Populate action_attrs map
        # outcome.action_attrs['action_attr1'] = 1.0 * (i + 1)
        # outcome.action_attrs['action_attr2'] = 2.0 * (i + 1)

        # # Set the reward value
        # outcome.reward = 10.0 * (i + 1)

        # # Populate outcome_attrs map
        # outcome.outcome_attrs['outcome_attr1'] = 5.0 * (i + 1)
        # outcome.outcome_attrs['outcome_attr2'] = 6.0 * (i + 1)
    else:
      for sample_id in self.complete_samples.keys():
        sample_details = self.complete_samples[sample_id]
        # print('sample_details : ', sample_details)
        outcome = response.outcome.add()
        outcome.reward = sample_details['reward']
        for k,v in sample_details['action'].items():
          outcome.action_attrs[k] = v
        for k,v in sample_details['outcome'].items():
          outcome.outcome_attrs[k] = v

    # print('response here: ', response)
    return response
    # raise SystemError


  @overrides
  def decision_point(
      self, request: service_pb2.DecisionPointRequest
  ) -> service_pb2.DecisionPointResponse:
    method_name = "decision_point"
    logging.debug(">>>>  In %s of %s", method_name, _file_name)

    self._lock.acquire()
    # todo : meetashah : add logic to fetch action stored from propose actions and send it as repsonse
    id, next_action = self.active_samples.popitem()
    self.worker_action_id_mapping[request.worker_id] = id
    self._lock.release()

    logging.info('next_action=%s', next_action)
    # raise SystemExit
    dp_response = service_pb2.DecisionPointResponse()
    dp_response.action.extend(param_dict_to_proto(next_action))
    logging.debug("<<<<  Out %s of %s", method_name, _file_name)
    dp_response.action_type = service_pb2.DecisionPointResponse.ActionType.AT_ACT
    return dp_response

  @overrides
  def finalize_episode(
      self, request: service_pb2.FinalizeEpisodeRequest
  ) -> service_pb2.FinalizeEpisodeResponse:
    method_name = "finalize_episode"
    logging.debug(">>>>  In %s of %s", method_name, _file_name)

    # logging.info("req in finalize episode of dummy.py : %s", request)
    self._lock.acquire()
    unique_action_id = self.worker_action_id_mapping[request.worker_id]
    self.complete_samples[
        unique_action_id
    ] = {
        'reward': request.decision_outcome.reward,
        # 'action': self.active_samples[unique_action_id],
        'action': param_proto_to_dict(request.decision_point.choice_params),
        'outcome': param_proto_to_dict(request.decision_outcome.outcome_params)
    }
    # self.complete_samples[
    #     unique_action_id-1
    # ] = {
    #     'reward': request.decision_outcome.reward+10,
    #     # 'action': self.active_samples[unique_action_id],
    #     'action': param_proto_to_dict(request.decision_point.choice_params),
    #     'outcome': param_proto_to_dict(request.decision_outcome.outcome_params)
    # }
    logging.info('FinalizeEpisode complete_samples=%s' % self.complete_samples)
    self._lock.release()

    # del self.active_samples[unique_action_id]
    # logging.info('FinalizeEpisode active_samples=%s' % self.active_samples)
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
        ' Progress"}\n'
    )
    self._lock.acquire()
    response += f'  #active_samples={len(self.active_samples)}\n'
    response += '  completed_samples=\n'
    response += ', '.join(list(self.actions)) + ', outcome\n'

    cur = [0] * len(self.actions)
    # action_keys = list(self.actions.keys())
    keys = sorted(self.actions.keys())
    logging.info('self.complete_samples=%s', self.complete_samples)

    reached_last = False
    while not reached_last:
      logging.info('cur(#%d)=%s', len(cur), cur)
      response += ', '.join(
          [str(self.possible_values[key][cur[i]]) for i, key in enumerate(keys)]
      )
      if tuple(cur) in self.complete_samples:
        response += ', ' + str(self.complete_samples[tuple(cur)]['outcome'])
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
