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

from concurrent import futures
import logging
import google.cloud.logging as log
from absl import app

import grpc
from dotenv import load_dotenv

load_dotenv()

import os
from overrides import overrides
import uuid

from typing import Any, Dict, List, Tuple
import time

from service import server_utils
from service import service_pb2
from service import service_pb2_grpc
from service.genetic_algorithm import GeneticAlgorithm
from service.exhaustive_search import ExhaustiveSearch
from service.llm import LLM
from service.optimizer_instance import OptimizerInstance
from service.vizier import Vizier
from service.acme_optimizer import Acme
from sight.proto import sight_pb2
from readerwriterlock import rwlock


_file_name = "server.py"
_resolve_times = []

instanceId = os.getenv("SPANNER_INSTANCE_ID")
databaseId = os.getenv("SPANNER_DATABASE_ID")
logtableId = os.getenv("SPANNER_LOG_TABLE_ID")
studytableId = os.getenv("SPANNER_STUDY_TABLE_ID")


def generate_unique_number() -> int:
  return uuid.uuid4().int & (1 << 63) - 1


def calculate_resolve_time(start_time):
  method_name = "calculate_resolve_time"
  logging.debug(">>>>>>>  In %s method of %s file.", method_name, _file_name)
  resolve_time = time.time() - start_time
  _resolve_times.append(resolve_time)
  avg_resolve_time = sum(_resolve_times) / len(_resolve_times)
  logging.debug(" logging.info : Average Resolve Time From Server: %s seconds",
               round(avg_resolve_time, 4))
  logging.debug("<<<<<< Out %s method of %s file.", method_name, _file_name)


class Optimizers:
  """
  Optimizer class to create request specific optimizer and use the methods
  provided in those to work with future requests.
  """

  def __init__(self):
    self.instances: Dict[str, OptimizerInstance] = {}
    self.instances_lock = rwlock.RWLockFair()

  def launch(
      self, request: service_pb2.LaunchRequest
  ) -> service_pb2.LaunchResponse:
    """Creates more specific optimizer and use them while responding to clients accordingly.
    """
    method_name = "launch"
    logging.debug(">>>>>>>  In %s method of %s file.", method_name, _file_name)
    optimizer_type = request.decision_config_params.optimizer_type
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
      elif optimizer_type == sight_pb2.DecisionConfigurationStart.OptimizerType.OT_ACME:
        self.instances[request.client_id] = Acme()
        obj = self.instances[request.client_id].launch(request)
        logging.info("self of optimizers class:  %s", str(self.__dict__))
        return obj
      elif optimizer_type == sight_pb2.DecisionConfigurationStart.OptimizerType.OT_LLM:
        self.instances[request.client_id] = LLM()
        return self.instances[request.client_id].launch(request)
      else:
        return service_pb2.LaunchResponse(
            display_string=f"OPTIMIZER '{optimizer_type}' NOT VALID!!"
        )
    logging.debug("<<<<<< Out %s method of %s file.", method_name, _file_name)


  def get_instance(self, client_id: str) -> OptimizerInstance:
    method_name = "get_instance"
    logging.debug(">>>>>>>  In %s method of %s file.", method_name, _file_name)
    with self.instances_lock.gen_rlock():
      instance_obj = self.instances[client_id]
      return instance_obj
    logging.debug("<<<<<< Out %s method of %s file.", method_name, _file_name)



class SightService(service_pb2_grpc.SightServiceServicer):
  """Service class to handle the grpc request send via sight client.
  """

  def __init__(self):
    super().__init__()
    self.optimizers = Optimizers()

  def Test(self, request, context):
    method_name = "Test"
    logging.debug(">>>>>>>  In %s method of %s file.", method_name, _file_name)
    obj = service_pb2.TestResponse(val="222")
    logging.debug("<<<<<< Out %s method of %s file.", method_name, _file_name)
    return obj

  def GetWeights(self, request, context):
    method_name = "GetWeights"
    logging.debug(">>>>>>>  In %s method of %s file.", method_name, _file_name)
    start_time = time.time()
    obj = self.optimizers.get_instance(request.client_id).get_weights(request)
    # calculate_resolve_time(start_time)
    logging.debug("<<<<<< Out %s method of %s file.", method_name, _file_name)
    return obj

  def DecisionPoint(self, request, context):
    method_name = "DecisionPoint"
    logging.debug(">>>>>>>  In %s method of %s file.", method_name, _file_name)
    start_time = time.time()
    obj = self.optimizers.get_instance(request.client_id).decision_point(
        request
    )
    # calculate_resolve_time(start_time)
    logging.debug("<<<<<<<  Out %s method of %s file.", method_name, _file_name)
    return obj

  def CurrentStatus(self, request, context):
    method_name = "CurrentStatus"
    logging.debug(">>>>>>>  In %s method of %s file.", method_name, _file_name)

    return self.optimizers.get_instance(request.client_id).current_status(
        request
    )
    logging.debug("<<<<<<<  Out %s method of %s file.", method_name, _file_name)

  def FetchOptimalAction(self, request, context):
    method_name = "FetchOptimalAction"
    logging.debug(">>>>>>>  In %s method of %s file.", method_name, _file_name)

    obj = self.optimizers.get_instance(request.client_id).fetch_optimal_action(
        request
    )
    logging.debug("<<<<<<<  Out %s method of %s file.", method_name, _file_name)
    return obj


  def ProposeAction(self, request, context):
    method_name = "ProposeAction"
    logging.debug(">>>>>>>  In %s method of %s file.", method_name, _file_name)

    return self.optimizers.get_instance(request.client_id).propose_action(
        request
    )
    logging.debug("<<<<<<<  Out %s method of %s file.", method_name, _file_name)

  def FinalizeEpisode(self, request, context):
    method_name = "FinalizeEpisode"
    logging.debug(">>>>>>>  In %s method of %s file.", method_name, _file_name)

    obj = self.optimizers.get_instance(request.client_id).finalize_episode(
        request
    )
    logging.debug("<<<<<<<  Out %s method of %s file.", method_name, _file_name)
    return obj

  def Launch(self, request, context):
    method_name = "Launch"
    logging.debug(">>>>>>>  In %s method of %s file.", method_name, _file_name)
    # start_time = time.time()
    logging.info("request here is : %s", request)
    obj = self.optimizers.launch(request)
    # calculate_resolve_time(start_time)
    logging.debug("<<<<<<<  Out %s method of %s file.", method_name, _file_name)
    return obj

  def Create(self, request, context):
    method_name = "Create"
    logging.debug(">>>>>>>  In %s method of %s file.", method_name, _file_name)
    # start_time = time.time()
    unique_id = generate_unique_number()
    # calculate_resolve_time(start_time)

    logging.debug("<<<<<<<  Out %s method of %s file.", method_name, _file_name)
    return service_pb2.CreateResponse(id=unique_id, path_prefix="/tmp/")


def serve():
  """Main method that listens on port 8080 and handle requests received from client.
  """
  method_name = "serve"
  logging.info(">>>>>>>  In %s method of %s file.", method_name, _file_name)

  server = grpc.server(futures.ThreadPoolExecutor(max_workers=500))
  service_pb2_grpc.add_SightServiceServicer_to_server(SightService(), server)
  server.add_insecure_port("[::]:8080")
  server.start()
  logging.info("server is up and running on port : 8080")

  server.wait_for_termination()
  logging.debug("<<<<<<<  Out %s method of %s file.", method_name, _file_name)

if __name__ == "__main__":
  method_name = "__main__"
  logging.basicConfig(level=logging.DEBUG)
  logging.debug(">>>>>>>  In %s method of %s file.", method_name, _file_name)
  try:
    app.run(serve())
  except BaseException as e:
    logging.error("Error occurred : ")
    logging.error(e)
  logging.debug("<<<<<<<  Out %s method of %s file.", method_name, _file_name)
