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
from readerwriterlock import rwlock
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
        self.last_sample = False
        self.exp_completed = False
        self.possible_values = {}
        self.max_reward_sample = {}
        self.pending_lock = rwlock.RWLockFair()
        self.active_lock = rwlock.RWLockFair()
        self.completed_lock = rwlock.RWLockFair()

    @overrides
    def launch(
            self,
            request: service_pb2.LaunchRequest) -> service_pb2.LaunchResponse:
        method_name = "launch"
        logging.debug(">>>>  In %s of %s", method_name, _file_name)
        response = super(WorklistScheduler, self).launch(request)
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

        with self.pending_lock.gen_wlock():
          self.pending_samples[self.unique_id] = [
              action_attrs, attributes
          ]

        # print('self.pending_samples : ',
        #       self.pending_samples)
        # print('self.active_samples : ',
        #       self.active_samples)
        # print('self.completed_samples : ',
        #       self.completed_samples)
        print('self.unique_id : ', self.unique_id)


        # Create response
        response = service_pb2.ProposeActionResponse(
            action_id=self.unique_id)
        self.unique_id += 1
        return response

    @overrides
    def GetOutcome(
        self, request: service_pb2.GetOutcomeRequest
    ) -> service_pb2.GetOutcomeResponse:
        # print('self.pending_samples : ',
        #       self.pending_samples)
        # print('self.active_samples : ',
        #       self.active_samples)
        # print('self.completed_samples : ',
        #       self.completed_samples)
        with self.completed_lock.gen_rlock():
          completed_samples = self.completed_samples
        with self.pending_lock.gen_rlock():
          pending_samples = self.pending_samples
        with self.active_lock.gen_rlock():
          active_samples = self.active_samples

        response = service_pb2.GetOutcomeResponse()
        if (request.unique_ids):
            required_samples = list(request.unique_ids)
            for sample_id in required_samples:
                outcome = response.outcome.add()
                outcome.action_id = sample_id
                if (sample_id in completed_samples):
                    sample_details = self.completed_samples[sample_id]
                    outcome.status = service_pb2.GetOutcomeResponse.Outcome.Status.COMPLETED
                    outcome.reward = sample_details['reward']
                    outcome.action_attrs.extend(param_dict_to_proto(
                        sample_details['action']))
                    outcome.outcome_attrs.extend(param_dict_to_proto(
                        sample_details['outcome']))
                    outcome.attributes.extend(param_dict_to_proto(
                      sample_details['attribute']))
                elif (sample_id in pending_samples):
                    outcome.status = service_pb2.GetOutcomeResponse.Outcome.Status.PENDING
                    outcome.response_str = '!! requested sample not yet assigned to any worker !!'
                elif any(value['id'] == sample_id for value in active_samples.values()):
                    outcome.status = service_pb2.GetOutcomeResponse.Outcome.Status.ACTIVE
                    outcome.response_str = '!! requested sample not completed yet !!'
                else:
                    outcome.status = service_pb2.GetOutcomeResponse.Outcome.Status.NOT_EXIST
                    outcome.response_str = f'!! requested sample Id {sample_id} does not exist !!'

                    print("!! NOT EXIST !!")
                    with self.active_lock.gen_rlock():
                      print(self.active_samples)
                    with self.pending_lock.gen_rlock():
                      print(self.pending_samples)
                    with self.completed_lock.gen_rlock():
                      print(self.completed_samples)
        else:
            for sample_id in completed_samples.keys():
                sample_details = completed_samples[sample_id]
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

        # print('self.pending_samples : ',
        #       self.pending_samples)
        # print('self.active_samples : ',
        #       self.active_samples)
        # print('self.completed_samples : ',
        #       self.completed_samples)
        # print('self.unique_id : ', self.unique_id)

        dp_response = service_pb2.DecisionPointResponse()
        # if(self.exp_completed):
        #   logging.info("sight experiment completed, killing the worker")
        #   dp_response.action_type = service_pb2.DecisionPointResponse.ActionType.AT_DONE
        # else:
        if self.pending_samples:

          # todo : meetashah : add logic to fetch action stored from propose actions and send it as repsonse
          # key, sample = self.pending_samples.popitem()
          # fetching the key in FIFO manner

          with self.pending_lock.gen_wlock():
            key = next(iter(self.pending_samples))
            sample = self.pending_samples.pop(key)

          with self.active_lock.gen_wlock():
            self.active_samples[request.worker_id] = {'id': key, 'sample': sample}


          next_action = sample[0]
          logging.info('next_action=%s', next_action)
          # raise SystemExit
          dp_response.action.extend(param_dict_to_proto(next_action))
          # print('self.active_samples : ', self.active_samples)
          # print('self.pending_samples : ', self.pending_samples)
          # print('self.completed_samples : ', self.completed_samples)
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

        with self.active_lock.gen_rlock():
          sample_dict = self.active_samples[request.worker_id]

        with self.completed_lock.gen_wlock():
          self.completed_samples[sample_dict['id']] = {
              # 'action': self.pending_samples[unique_action_id],
              'action':
              param_proto_to_dict(request.decision_point.choice_params),
              'attribute': sample_dict['sample'][1],
              'reward': request.decision_outcome.reward,
              'outcome':
              param_proto_to_dict(request.decision_outcome.outcome_params)
          }

        with self.active_lock.gen_wlock():
          del self.active_samples[request.worker_id]


        # print('self.active_samples : ', self.active_samples)
        # print('self.pending_samples : ', self.pending_samples)
        # print('self.completed_samples : ', self.completed_samples)
        logging.debug("<<<<  Out %s of %s", method_name, _file_name)
        return service_pb2.FinalizeEpisodeResponse(response_str='Success!')

    @overrides
    def current_status(
        self, request: service_pb2.CurrentStatusRequest
    ) -> service_pb2.CurrentStatusResponse:
        method_name = "current_status"
        logging.debug(">>>>  In %s of %s", method_name, _file_name)
        # add logic to check status - ref from exhaustive search

    @overrides
    def fetch_optimal_action(
        self, request: service_pb2.FetchOptimalActionRequest
    ) -> service_pb2.FetchOptimalActionResponse:
        method_name = "fetch_optimal_action"
        logging.debug(">>>>  In %s of %s", method_name, _file_name)
        # add logic to check status - ref from exhaustive search
        logging.debug("<<<<  Out %s of %s", method_name, _file_name)

    @overrides
    def close(
       self, request: service_pb2.CloseRequest
    ) -> service_pb2.CloseResponse:
        method_name = "close"
        logging.debug(">>>>  In %s of %s", method_name, _file_name)
        self.exp_completed = True
        print("sight experiment completed....")
        # logging.debug("************************closed*******************************")
        logging.debug("<<<<  Out %s of %s", method_name, _file_name)
        return service_pb2.CloseResponse(response_str="success")

    @overrides
    def WorkerAlive(
       self, request: service_pb2.WorkerAliveRequest
    ) -> service_pb2.WorkerAliveResponse:
        method_name = "WorkerAlive"
        logging.debug(">>>>  In %s of %s", method_name, _file_name)
        if(self.exp_completed):
           worker_alive_status = False
        else:
           worker_alive_status = True
        logging.debug("<<<<  Out %s of %s", method_name, _file_name)
        return service_pb2.WorkerAliveResponse(status=worker_alive_status)

