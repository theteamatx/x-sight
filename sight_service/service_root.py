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

from absl import app
from absl import flags
from concurrent import futures
from collections import defaultdict
from dotenv import load_dotenv
import functools
import grpc
import logging
import math

load_dotenv()

import os
import time
from typing import Any, Dict, List, Tuple
import uuid
import sys

# from overrides import overrides
from readerwriterlock import rwlock
from sight.proto import sight_pb2
# from sight_service import service_utils
# from sight_service.acme_optimizer import Acme
from sight_service.bayesian_opt import BayesianOpt
from sight_service.exhaustive_search import ExhaustiveSearch
from sight_service.genetic_algorithm import GeneticAlgorithm
from sight_service.llm import LLM
from sight_service.nevergrad_opt import NeverGradOpt
from sight_service.optimizer_instance import OptimizerInstance
from sight_service.proto import service_pb2
from sight_service.proto import service_pb2_grpc
from sight_service.sensitivity_analysis import SensitivityAnalysis
from sight_service.smc_py import SMCPy
from sight_service.vizier import Vizier
from sight_service.worklist_scheduler_opt import WorklistScheduler

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
    logging.info(f"<<<<<< {func.__name__}, file {os.path.basename(__file__)} with args={args}")
    
    if 'request' in kwargs:
      if 'client_id' in kwargs['request'].keys():
        if kwargs['request'].client_id == 0:
          raise ValueError(f'Empty log identifier in {func.__name__}.')

    start_time = time.time()
    result = func(*args, **kwargs)
    elapsed_time = time.time() - start_time
    func_to_elapsed_time[func.__name__] += elapsed_time
    func_to_elapsed_time_sq[func.__name__] += elapsed_time*elapsed_time
    func_call_count[func.__name__] += 1

    mean = func_to_elapsed_time[func.__name__]/func_call_count[func.__name__]
    mean_sq = func_to_elapsed_time_sq[func.__name__]/func_call_count[func.__name__]

    logging.info('>>>>>> %s, file %s, elapsed: (this=%f, avg=%f, rel_sd=%f, count=%d)', 
                 func.__name__,
                 os.path.basename(__file__),
                 elapsed_time, 
                 mean,
                 math.sqrt(mean_sq - mean*mean)/mean if mean != 0 else 0,
                 func_call_count[func.__name__],
                 )
    return result
  return wrapper

def calculate_resolve_time(start_time):
  logging.info(">>>>>>>  In %s method of %s file.", sys._getframe().f_code.co_name, os.path.basename(__file__))
  resolve_time = time.time() - start_time
  _resolve_times.append(resolve_time)
  avg_resolve_time = sum(_resolve_times) / len(_resolve_times)
  logging.info(" logging.info : Average Resolve Time From Server: %s seconds",
               round(avg_resolve_time, 4))
  logging.info("<<<<<< Out %s method of %s file.", sys._getframe().f_code.co_name, os.path.basename(__file__))


class Optimizers:
  """
  Optimizer class to create request specific optimizer and use the methods
  provided in those to work with future requests.
  """

  def __init__(self):
    self.instances: Dict[str, OptimizerInstance] = {}
    self.instances_lock = rwlock.RWLockFair()

  def launch(self,
             request: service_pb2.LaunchRequest) -> service_pb2.LaunchResponse:
    """Creates more specific optimizer and use them while responding to clients accordingly.
    """
    optimizer_type = request.decision_config_params.optimizer_type
    logging.debug(">>>>>>>  In %s method of %s file. optimizer_type=%s",
                  sys._getframe().f_code.co_name, os.path.basename(__file__), optimizer_type)
    with self.instances_lock.gen_wlock():
      if optimizer_type == sight_pb2.DecisionConfigurationStart.OptimizerType.OT_VIZIER:
        self.instances[request.client_id] = Vizier()
        return self.instances[request.client_id].launch(request)
      elif optimizer_type == sight_pb2.DecisionConfigurationStart.OptimizerType.OT_GENETIC_ALGORITHM:
        self.instances[request.client_id] = GeneticAlgorithm()
        return self.instances[request.client_id].launch(request)
      elif optimizer_type == sight_pb2.DecisionConfigurationStart.OptimizerType.OT_EXHAUSTIVE_SEARCH:
        self.instances[request.client_id] = ExhaustiveSearch()
        return self.instances[request.client_id].launch(request)
      # elif optimizer_type == sight_pb2.DecisionConfigurationStart.OptimizerType.OT_ACME:
      #   self.instances[request.client_id] = Acme()
      #   obj = self.instances[request.client_id].launch(request)
      #   # logging.info("self of optimizers class:  %s", str(self.__dict__))
      #   return obj
      elif optimizer_type == sight_pb2.DecisionConfigurationStart.OptimizerType.OT_LLM:
        self.instances[request.client_id] = LLM()
        obj = self.instances[request.client_id].launch(request)
        return obj
      elif optimizer_type == sight_pb2.DecisionConfigurationStart.OptimizerType.OT_BAYESIAN_OPT:
        self.instances[request.client_id] = BayesianOpt()
        obj = self.instances[request.client_id].launch(request)
        return obj
      elif optimizer_type == sight_pb2.DecisionConfigurationStart.OptimizerType.OT_SENSITIVITY_ANALYSIS:
        self.instances[request.client_id] = SensitivityAnalysis()
        obj = self.instances[request.client_id].launch(request)
        return obj
      elif optimizer_type == sight_pb2.DecisionConfigurationStart.OptimizerType.OT_NEVER_GRAD:
        self.instances[request.client_id] = NeverGradOpt()
        obj = self.instances[request.client_id].launch(request)
        return obj
      elif optimizer_type == sight_pb2.DecisionConfigurationStart.OptimizerType.OT_SMC_PY:
        self.instances[request.client_id] = SMCPy()
        obj = self.instances[request.client_id].launch(request)
        return obj
      elif optimizer_type == sight_pb2.DecisionConfigurationStart.OptimizerType.OT_WORKLIST_SCHEDULER:
        self.instances[request.client_id] = WorklistScheduler()
        obj = self.instances[request.client_id].launch(request)
        return obj
      else:
        return service_pb2.LaunchResponse(
            display_string=f"OPTIMIZER '{optimizer_type}' NOT VALID!!")
      
    logging.info("<<<<<< Out %s method of %s file.", sys._getframe().f_code.co_name, os.path.basename(__file__))

  def get_instance(self, client_id: str) -> OptimizerInstance:
    # logging.debug(">>>>>>>  In %s method of %s file.", sys._getframe().f_code.co_name, os.path.basename(__file__))
    with self.instances_lock.gen_rlock():
      if (client_id in self.instances):
        instance_obj = self.instances[client_id]
        return instance_obj
      else:
        #add better mechanism, this require in close rpc for now
        return None
    # logging.debug("<<<<<< Out %s method of %s file.", sys._getframe().f_code.co_name, os.path.basename(__file__))


class SightService(service_pb2_grpc.SightServiceServicer):
  """Service class to handle the grpc request send via sight client.
  """

  def __init__(self):
    super().__init__()
    self.optimizers = Optimizers()
    logging.info('SightService::__init__')


  @rpc_call
  def Test(self, request, context):
    return service_pb2.TestResponse(val="222")

  # def GetWeights(self, request, context):
  #   logging.info(">>>>>>>  In %s method of %s file.", sys._getframe().f_code.co_name, os.path.basename(__file__))
  #   start_time = time.time()
  #   obj = self.optimizers.get_instance(request.client_id).get_weights(request)
  #   # calculate_resolve_time(start_time)
  #   logging.info("<<<<<< Out %s method of %s file.", sys._getframe().f_code.co_name, os.path.basename(__file__))
  #   return obj

  @rpc_call
  def DecisionPoint(self, request, context):
    return self.optimizers.get_instance(
        request.client_id).decision_point(request)

  @rpc_call
  def Tell(self, request, context):
    return self.optimizers.get_instance(request.client_id).tell(request)

  @rpc_call
  def Listen(self, request, context):
    return self.optimizers.get_instance(request.client_id).listen(request)

  @rpc_call
  def CurrentStatus(self, request, context):
    return self.optimizers.get_instance(
        request.client_id).current_status(request)

  @rpc_call
  def FetchOptimalAction(self, request, context):
    return self.optimizers.get_instance(
        request.client_id).fetch_optimal_action(request)

  @rpc_call
  def ProposeAction(self, request, context):
    return self.optimizers.get_instance(
        request.client_id).propose_action(request)

  @rpc_call
  def GetOutcome(self, request, context):
    return self.optimizers.get_instance(request.client_id).GetOutcome(request)

  @rpc_call
  def FinalizeEpisode(self, request, context):
    return self.optimizers.get_instance(
        request.client_id).finalize_episode(request)

  @rpc_call
  def Launch(self, request, context):
    return self.optimizers.launch(request)

  @rpc_call
  def Create(self, request, context):
    unique_id = generate_unique_number()
    return service_pb2.CreateResponse(id=unique_id, path_prefix="/tmp/")

  @rpc_call
  def Close(self, request, context):
    # only call if it's launch called, otherwise no entry of opt for that client
    if (self.optimizers.get_instance(request.client_id)):
      obj = self.optimizers.get_instance(request.client_id).close(request)
    else:
      obj = service_pb2.CloseResponse()
    #? do we need to remove entry from optimizer dict, if available??
    return obj

  @rpc_call
  def WorkerAlive(self, request, context):
    return self.optimizers.get_instance(request.client_id).WorkerAlive(request)


def serve():
  """Main method that listens on port 8080 and handle requests received from client.
    """
  logging.info(">>>>>>>  In %s method of %s file.", sys._getframe().f_code.co_name, os.path.basename(__file__))

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
  logging.info("<<<<<<<  Out %s method of %s file.", sys._getframe().f_code.co_name, os.path.basename(__file__))


def main(argv):
  logging.basicConfig(level=logging.INFO)
  logging.info(">>>>>>>  In %s method of %s file.", sys._getframe().f_code.co_name, os.path.basename(__file__))
  try:
    app.run(serve())
  except BaseException as e:
    logging.error("Error occurred : ")
    logging.error(e)
  logging.info("<<<<<<<  Out %s method of %s file.", sys._getframe().f_code.co_name, os.path.basename(__file__))


if __name__ == "__main__":
  app.run(main)
