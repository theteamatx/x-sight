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

import threading

from bayes_opt import BayesianOptimization
from bayes_opt import UtilityFunction
from helpers.logs.logs_handler import logger as logging
from overrides import overrides
from sight.proto import sight_pb2
from sight.utils.proto_conversion import convert_dict_to_proto
from sight.utils.proto_conversion import convert_proto_to_dict
from sight_service.optimizer_instance import OptimizerInstance
from sight_service.proto import service_pb2

_file_name = "bayesian_opt.py"


class BayesianOpt(OptimizerInstance):
  """Uses an LLM to choose the parameters of the code.
  """

  def __init__(self):
    super().__init__()
    self._lock = threading.RLock()
    self._total_count = 0
    self._completed_count = 0
    self.function_code = None

  @overrides
  def launch(self,
             request: service_pb2.LaunchRequest) -> service_pb2.LaunchResponse:
    response = super(BayesianOpt, self).launch(request)
    self._total_count = request.decision_config_params.num_trials
    self._optimizer = BayesianOptimization(
        f=None,
        pbounds={
            key: (p.min_value, p.max_value) for key, p in self.actions.items()
        },
        verbose=2,
        allow_duplicate_points=True,
        # random_state=1,
    )
    self._utility = UtilityFunction(kind='ucb', kappa=1.96, xi=0.01)
    response.display_string = 'BayesianOpt Start'
    return response

  # def _params_to_dict(self, dp: sight_pb2) -> Dict[str, float]:
  #   """Returns the dict representation of a DecisionParams proto"""
  #   d = {}
  #   for a in dp:
  #     d[a.key] = a.value.double_value
  #   return d

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

    dp_response.action.CopyFrom(convert_dict_to_proto(dict=selected_actions))

    print('DecisionPoint response=%s' % dp_response)
    dp_response.action_type = service_pb2.DecisionPointResponse.ActionType.AT_ACT
    return dp_response

  @overrides
  def finalize_episode(
      self, request: service_pb2.FinalizeEpisodeRequest
  ) -> service_pb2.FinalizeEpisodeResponse:
    # logging.info('FinalizeEpisode request=%s', request)

    self._lock.acquire()
    for i in range(len(request.decision_messages)):
      d = convert_proto_to_dict(proto=request.decision_messages[i].decision_point.choice_params)
      logging.info('FinalizeEpisode outcome=%s / %s',
                  request.decision_messages[i].decision_outcome.reward, d)
      # d = {}
      # for a in request.decision_point.choice_params:
      #   d[a.key] = a.value.double_value
      self._optimizer.register(params=d, target=request.decision_messages[i].decision_outcome.reward)
      # self._completed_count += 1
    self._lock.release()
    return service_pb2.FinalizeEpisodeResponse(response_str='Success!')

  @overrides
  def current_status(
      self, request: service_pb2.CurrentStatusRequest
  ) -> service_pb2.CurrentStatusResponse:
    output = '[BayesianOpt (#%s trials)\n' % len(self._optimizer.res)
    for trial in sorted(self._optimizer.res,
                        key=lambda x: x['target'],
                        reverse=True):
      output += '   ' + str(trial) + '\n'
    output += ']\n'

    if (self._completed_count == self._total_count):
      status = service_pb2.CurrentStatusResponse.Status.SUCCESS
    elif (self._completed_count < self._total_count):
      status = service_pb2.CurrentStatusResponse.Status.IN_PROGRESS
    else:
      status = service_pb2.CurrentStatusResponse.Status.FAILURE

    return service_pb2.CurrentStatusResponse(response_str=output, status=status)

  @overrides
  def WorkerAlive(
      self, request: service_pb2.WorkerAliveRequest
  ) -> service_pb2.WorkerAliveResponse:
    method_name = "WorkerAlive"
    logging.debug(">>>>  In %s of %s", method_name, _file_name)
    response = service_pb2.WorkerAliveResponse()
    if (self._completed_count == self._total_count):
      response.status_type = service_pb2.WorkerAliveResponse.StatusType.ST_DONE
    elif(not self.function_code):
       response.status_type = service_pb2.WorkerAliveResponse.StatusType.ST_RETRY
    else:
      # Increasing count here so that multiple workers can't enter the dp call for same sample at last
      self._completed_count += 1

      self._lock.acquire()
      selected_actions = self._optimizer.suggest(self._utility)
      self._lock.release()

      decision_message = response.decision_messages.add()
      decision_message.action_id = self._completed_count
      decision_message.action.CopyFrom(convert_dict_to_proto(dict=selected_actions))
      response.function_code = self.function_code

      response.status_type = service_pb2.WorkerAliveResponse.StatusType.ST_ACT

    logging.info("worker_alive_status is %s", response.status_type)
    logging.debug("<<<<  Out %s of %s", method_name, _file_name)
    return response

  @overrides
  def close(self,
            request: service_pb2.CloseRequest) -> service_pb2.CloseResponse:
    logging.info(
        "sight experiment completed in bayes_opt")

    return service_pb2.CloseResponse(response_str="success")

  def send_function(self, request: service_pb2.SendFunctionRequest) -> service_pb2.SendFunctionResponse:
    self.function_code = request.function_code
    return service_pb2.SendFunctionResponse(response_str="success")
