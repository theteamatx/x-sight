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
"""The Python implementation of the GRPC decision.SightService server."""


def warn(*args, **kwargs):
  pass


import warnings

warnings.warn = warn

from collections import defaultdict
from concurrent import futures
import functools
import logging
import math

from absl import app
from absl import flags
from dotenv import load_dotenv
import grpc

load_dotenv()

from collections import defaultdict
import os
import sys
import time
from typing import Any, Dict, List, Optional, Tuple, Union
import uuid
import datetime

from readerwriterlock import rwlock
# from sight_service import service_utils
from sight.proto import sight_pb2
# from sight_service.acme_optimizer import Acme
from sight_service.bayesian_opt import BayesianOpt
from sight_service.exhaustive_search import ExhaustiveSearch
from sight_service.genetic_algorithm import GeneticAlgorithm
from sight_service.llm import LLM
from sight_service.nevergrad_opt import NeverGradOpt
from sight_service.optimizer_instance import OptimizerInstance #not used anymore, replaced with WorkerSyncContext
from sight_service.worker_sync_context import WorkerSyncContext
from sight_service import worker_sync_handler
from sight_service.proto import service_pb2
from sight_service.proto import service_pb2_grpc
from sight_service.sensitivity_analysis import SensitivityAnalysis
from sight_service.smc_py import SMCPy
from sight_service.vizier import Vizier
from sight_service.worklist_scheduler_opt import WorklistScheduler
from sight_service.message_queue.message_logger.interface import (
    ILogStorageCollectStrategy
)
from sight_service.message_queue.message_logger.log_storage_collect import (
    CachedBasedLogStorageCollectStrategy
)
from sight_service.message_queue.queue_factory import queue_factory
datetime = datetime.datetime


_PORT = flags.DEFINE_integer('port', 8080, 'The port to listen on')
_resolve_times = []

instanceId = os.getenv("SPANNER_INSTANCE_ID")
databaseId = os.getenv("SPANNER_DATABASE_ID")
logtableId = os.getenv("SPANNER_LOG_TABLE_ID")
studytableId = os.getenv("SPANNER_STUDY_TABLE_ID")


def generate_unique_number() -> int:
  return uuid.uuid4().int & (1 << 63) - 1


import logging

func_to_elapsed_time = defaultdict(float)
func_to_elapsed_time_sq = defaultdict(float)
func_call_count = defaultdict(float)


def rpc_call(func):

  @functools.wraps(func)
  def wrapper(*args, **kwargs):
    logging.debug(
        f"<<<<<< {func.__name__}, file {os.path.basename(__file__)} with args={args}"
    )

    if 'request' in kwargs:
      if 'client_id' in kwargs['request'].keys():
        if kwargs['request'].client_id == 0:
          raise ValueError(f'Empty log identifier in {func.__name__}.')

    start_time = time.time()
    result = func(*args, **kwargs)
    elapsed_time = time.time() - start_time
    func_to_elapsed_time[func.__name__] += elapsed_time
    func_to_elapsed_time_sq[func.__name__] += elapsed_time * elapsed_time
    func_call_count[func.__name__] += 1

    mean = func_to_elapsed_time[func.__name__] / func_call_count[func.__name__]
    mean_sq = func_to_elapsed_time_sq[func.__name__] / func_call_count[
        func.__name__]

    logging.debug(
        '>>>>>> %s, file %s, elapsed: (this=%f, avg=%f, rel_sd=%f, count=%d)',
        func.__name__,
        os.path.basename(__file__),
        elapsed_time,
        mean,
        math.sqrt(mean_sq - mean * mean) / mean if mean != 0 else 0,
        func_call_count[func.__name__],
    )
    return result

  return wrapper

# class Optimizers:
class WorkerSyncManager:
  """
  WorkerSyncManager class to manage sync up between different worker of same
  experiment.
  """

  def __init__(self):
    self.instances: Dict[str, Dict[str, WorkerSyncContext]] = defaultdict(dict)
    self.instances_lock = rwlock.RWLockFair()

  def launch(self,
             request: service_pb2.LaunchRequest) -> service_pb2.LaunchResponse:
    """Creates more specific optimizer and use them while responding to clients accordingly.
    """
    optimizer_type = request.decision_config_params.optimizer_type
    mq_batch_size = request.decision_config_params.server_queue_batch_size
    cache_mode = request.decision_config_params.cache_mode
    logging.debug(">>>>>>>  In %s method of %s file. optimizer_type=%s",
                  sys._getframe().f_code.co_name, os.path.basename(__file__),
                  optimizer_type)
    with self.instances_lock.gen_wlock():
      self.instances[request.client_id][
            request.question_label] = WorkerSyncContext(queue=queue_factory(
                    queue_type="shared_lock_list",
                    batch_size=mq_batch_size,
                    logger_storage_strategy=None,
                ))

    return service_pb2.LaunchResponse(
       display_string=f"queue created successfully....")

      # if optimizer_type == sight_pb2.DecisionConfigurationStart.OptimizerType.OT_VIZIER:
      #   self.instances[request.client_id][request.question_label] = Vizier()
      #   return self.instances[request.client_id][request.question_label].launch(
      #       request)
      # elif optimizer_type == sight_pb2.DecisionConfigurationStart.OptimizerType.OT_GENETIC_ALGORITHM:
      #   self.instances[request.client_id][
      #       request.question_label] = GeneticAlgorithm()
      #   return self.instances[request.client_id][request.question_label].launch(
      #       request)
      # elif optimizer_type == sight_pb2.DecisionConfigurationStart.OptimizerType.OT_EXHAUSTIVE_SEARCH:
      #   self.instances[request.client_id][
      #       request.question_label] = ExhaustiveSearch()
      #   return self.instances[request.client_id][request.question_label].launch(
      #       request)
      # # elif optimizer_type == sight_pb2.DecisionConfigurationStart.OptimizerType.OT_ACME:
      # #   self.instances[request.client_id][request.question_label] = Acme()
      # #   obj = self.instances[request.client_id][request.question_label].launch(request)
      # #   # logging.info("self of optimizers class:  %s", str(self.__dict__))
      # #   return obj
      # elif optimizer_type == sight_pb2.DecisionConfigurationStart.OptimizerType.OT_LLM:
      #   self.instances[request.client_id][request.question_label] = LLM()
      #   obj = self.instances[request.client_id][request.question_label].launch(
      #       request)
      #   return obj
      # elif optimizer_type == sight_pb2.DecisionConfigurationStart.OptimizerType.OT_BAYESIAN_OPT:
      #   self.instances[request.client_id][
      #       request.question_label] = BayesianOpt()
      #   obj = self.instances[request.client_id][request.question_label].launch(
      #       request)
      #   return obj
      # elif optimizer_type == sight_pb2.DecisionConfigurationStart.OptimizerType.OT_SENSITIVITY_ANALYSIS:
      #   self.instances[request.client_id][
      #       request.question_label] = SensitivityAnalysis()
      #   obj = self.instances[request.client_id][request.question_label].launch(
      #       request)
      #   return obj
      # elif optimizer_type == sight_pb2.DecisionConfigurationStart.OptimizerType.OT_NEVER_GRAD:
      #   self.instances[request.client_id][
      #       request.question_label] = NeverGradOpt()
      #   obj = self.instances[request.client_id][request.question_label].launch(
      #       request)
      #   return obj
      # elif optimizer_type == sight_pb2.DecisionConfigurationStart.OptimizerType.OT_SMC_PY:
      #   self.instances[request.client_id][request.question_label] = SMCPy()
      #   obj = self.instances[request.client_id][request.question_label].launch(
      #       request)
      #   return obj
      # elif optimizer_type == sight_pb2.DecisionConfigurationStart.OptimizerType.OT_WORKLIST_SCHEDULER:
      #   self.instances[request.client_id][
      #       request.question_label] = WorklistScheduler(meta_data={
      #           "mq_batch_size": mq_batch_size,
      #           "cache_mode": cache_mode
      #       })
      #   obj = self.instances[request.client_id][request.question_label].launch(
      #       request)
      #   return obj
      # else:
      #   return service_pb2.LaunchResponse(
      #       display_string=f"OPTIMIZER '{optimizer_type}' NOT VALID!!")

    logging.debug("<<<<<< Out %s method of %s file.",
                  sys._getframe().f_code.co_name, os.path.basename(__file__))

  def get_instance(
      self,
      client_id: str,
      question: Optional[str] = None) -> Union[WorkerSyncContext, None]:
    with self.instances_lock.gen_rlock():
      if client_id in self.instances:
        if question is None:
          # Return all instances for this client_id
          return self.instances[client_id]
        return self.instances[client_id].get(question)
      return None


class SightService(service_pb2_grpc.SightServiceServicer):
  """Service class to handle the grpc request send via sight client.
  """

  def __init__(self):
    super().__init__()
    # self.optimizers = Optimizers()
    self.worker_sync_manager = WorkerSyncManager()
    logging.debug('SightService::__init__')

  @rpc_call
  def Test(self, request, context):
    method_name = "Test"
    logging.info(">>>>>>>  In %s method of %s file.", method_name,
                 os.path.basename(__file__))
    obj = service_pb2.TestResponse()
    obj.val = str(222)
    logging.info("<<<<<< Out %s method of %s file.", method_name,
                 os.path.basename(__file__))
    return obj

  # def GetWeights(self, request, context):
  #   logging.debug(">>>>>>>  In %s method of %s file.", sys._getframe().f_code.co_name, os.path.basename(__file__))
  #   start_time = time.time()
  #   obj = self.optimizers.get_instance(request.client_id).get_weights(request)
  #   # calculate_resolve_time(start_time)
  #   logging.debug("<<<<<< Out %s method of %s file.", sys._getframe().f_code.co_name, os.path.basename(__file__))
  #   return obj

  # deprecated - not used anymore
  # @rpc_call
  # def DecisionPoint(self, request, context):
  #   return self.optimizers.get_instance(
  #       request.client_id, request.question_label).decision_point(request)

  @rpc_call
  def Tell(self, request, context):
    return self.optimizers.get_instance(request.client_id,
                                        request.question_label).tell(request)

  @rpc_call
  def Listen(self, request, context):
    return self.optimizers.get_instance(request.client_id,
                                        request.question_label).listen(request)

  @rpc_call
  def CurrentStatus(self, request, context):
    return self.optimizers.get_instance(
        request.client_id, request.question_label).current_status(request)

  @rpc_call
  def FetchOptimalAction(self, request, context):
    return self.optimizers.get_instance(
        request.client_id, request.question_label).fetch_optimal_action(request)

  @rpc_call
  def ProposeAction(self, request, context):
    logging.info('request=%s', request)
    # return self.optimizers.get_instance(
    #     request.client_id, request.question_label).propose_action(request)
    sync_ctx = self.worker_sync_manager.get_instance(
          request.client_id, request.question_label
      )
    return worker_sync_handler.handle_propose_action(sync_ctx, request)

  @rpc_call
  def GetOutcome(self, request, context):
    # return self.optimizers.get_instance(
    #     request.client_id, request.question_label).GetOutcome(request)
    sync_ctx = self.worker_sync_manager.get_instance(
          request.client_id, request.question_label
      )
    return worker_sync_handler.handle_get_outcome(sync_ctx, request)

  @rpc_call
  def FinalizeEpisode(self, request, context):
    # return self.optimizers.get_instance(
    #     request.client_id, request.question_label).finalize_episode(request)
    sync_ctx = self.worker_sync_manager.get_instance(
          request.client_id, request.question_label
      )
    return worker_sync_handler.handle_finalize_episode(sync_ctx, request)

  @rpc_call
  def Launch(self, request, context):
    return self.worker_sync_manager.launch(request)

  @rpc_call
  def Create(self, request, context):
    unique_id = generate_unique_number()
    return service_pb2.CreateResponse(id=unique_id, path_prefix="/tmp/")

  @rpc_call
  def Close(self, request, context):

    logging.info('request in close is : %s', request)
    with self.worker_sync_manager.instances_lock.gen_rlock():
      # fixed now - there is an issue with this : even one of the worker calls the close,
      # this will call the close on the optimizer - need to fix this
      # logging.info("request => %s", request)
      # if request.HasField("question_label"):
      #   instance = self.worker_sync_manager.get_instance(request.client_id,
      #                                            request.question_label)
      #   # print('*********lenght of instances : ', len(instances))
      #   if instance:
      #   #   for question, obj in instances.items():
      #     # logging.info('instance found : %s', instance)
      #     obj = instance.close(request)
      #   else:
      #     logging.info(
      #         "client id not present in server, no launch ever called for this client??"
      #     )
      #     obj = service_pb2.CloseResponse()
      # else:
      #   logging.info(
      #       "root process close called"
      #   )
      #   obj = service_pb2.CloseResponse()

      if not request.HasField("question_label"):
        logging.info("root process close called")
        obj = service_pb2.CloseResponse()
        return obj

      instance = self.worker_sync_manager.get_instance(request.client_id,
                                              request.question_label)

      if not instance:
        logging.info(
            "client id not present in server, no launch ever called for this client??"
        )
        obj = service_pb2.CloseResponse()
        return obj

      # obj = instance.close(request)
      obj = service_pb2.CloseResponse()
      return obj

    #? do we need to remove entry from optimizer dict, if available??
    return obj

  @rpc_call
  def WorkerAlive(self, request, context):
    logging.info('called worker_alive for lable %s', request.question_label)
    # return self.optimizers.get_instance(
    #     request.client_id, request.question_label).WorkerAlive(request)
    sync_ctx = self.worker_sync_manager.get_instance(
          request.client_id, request.question_label
      )
    return worker_sync_handler.handle_worker_alive(sync_ctx, request)


def serve():
  """Main method that listens on port 8080 and handle requests received from client.
    """
  logging.info(">>>>>>>  In %s method of %s file.",
               sys._getframe().f_code.co_name, os.path.basename(__file__))

  server = grpc.server(futures.ThreadPoolExecutor(max_workers=500),
                       options=[
                           ('grpc.max_receive_message_length',
                            512 * 1024 * 1024),
                       ])
  service_pb2_grpc.add_SightServiceServicer_to_server(SightService(), server)
  server.add_insecure_port(f"[::]:{_PORT.value}")
  server.start()
  logging.info(f"server is up and running on port : {_PORT.value}")

  # flask_app.run(debug=True, host="0.0.0.0", port=_PORT.value)
  server.wait_for_termination()
  logging.info("<<<<<<<  Out %s method of %s file.",
               sys._getframe().f_code.co_name, os.path.basename(__file__))


def main(argv):
  logging.basicConfig(level=logging.INFO)
  logging.info(">>>>>>>  In %s method of %s file.",
               sys._getframe().f_code.co_name, os.path.basename(__file__))
  try:
    app.run(serve())
  except BaseException as e:
    logging.error("Error occurred : ")
    logging.error(e)
  logging.info("<<<<<<<  Out %s method of %s file.",
               sys._getframe().f_code.co_name, os.path.basename(__file__))


if __name__ == "__main__":
  app.run(main)
