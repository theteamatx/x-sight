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

from sight_service import service_utils
from sight.proto import sight_pb2
from sight_service.acme_optimizer import Acme
from sight_service.bayesian_opt import BayesianOpt
from sight_service.exhaustive_search import ExhaustiveSearch
from sight_service.genetic_algorithm import GeneticAlgorithm
from sight_service.llm import LLM
from sight_service.nevergrad_opt import NeverGradOpt
from sight_service.optimizer_instance import OptimizerInstance
from sight_service.proto import service_pb2
from sight_service.proto import service_pb2_grpc
from sight_service.sensitivity_analysis import SensitivityAnalysis
from sight_service.vizier import Vizier
from readerwriterlock import rwlock

from flask import Flask

flask_app = Flask(__name__)

@flask_app.route("/")
def hello_world():
    return "<p>Root!</p>"

_file_name = "service_root.py"
_resolve_times = []

instanceId = os.getenv("SPANNER_INSTANCE_ID")
databaseId = os.getenv("SPANNER_DATABASE_ID")
logtableId = os.getenv("SPANNER_LOG_TABLE_ID")
studytableId = os.getenv("SPANNER_STUDY_TABLE_ID")


def generate_unique_number() -> int:
  return uuid.uuid4().int & (1 << 63) - 1


def calculate_resolve_time(start_time):
  method_name = "calculate_resolve_time"
  logging.info(">>>>>>>  In %s method of %s file.", method_name, _file_name)
  resolve_time = time.time() - start_time
  _resolve_times.append(resolve_time)
  avg_resolve_time = sum(_resolve_times) / len(_resolve_times)
  logging.info(" logging.info : Average Resolve Time From Server: %s seconds",
               round(avg_resolve_time, 4))
  logging.info("<<<<<< Out %s method of %s file.", method_name, _file_name)


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
    logging.info(">>>>>>>  In %s method of %s file.", method_name, "Optimizers")

    optimizer_type = request.decision_config_params.optimizer_type
    logging.debug(">>>>>>>  In %s method of %s file. optimizer_type=%s", method_name, _file_name, optimizer_type)
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
        # logging.info("self of optimizers class:  %s", str(self.__dict__))
        return obj
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
      else:
        return service_pb2.LaunchResponse(
            display_string=f"OPTIMIZER '{optimizer_type}' NOT VALID!!"
        )
    logging.info("<<<<<< Out %s method of %s file.", method_name, _file_name)


  def get_instance(self, client_id: str) -> OptimizerInstance:
    method_name = "get_instance"
    logging.info(">>>>>>>  In %s method of %s file.", method_name, _file_name)
    with self.instances_lock.gen_rlock():
      instance_obj = self.instances[client_id]
      return instance_obj
    logging.info("<<<<<< Out %s method of %s file.", method_name, _file_name)



class SightService(service_pb2_grpc.SightServiceServicer):
  """Service class to handle the grpc request send via sight client.
  """

  def __init__(self):
    super().__init__()
    self.optimizers = Optimizers()

  def Test(self, request, context):
    method_name = "Test"
    logging.info(">>>>>>>  In %s method of %s file.", method_name, _file_name)
    obj = service_pb2.TestResponse(val="222")
    logging.info("<<<<<< Out %s method of %s file.", method_name, _file_name)
    return obj

  # def GetWeights(self, request, context):
  #   method_name = "GetWeights"
  #   logging.info(">>>>>>>  In %s method of %s file.", method_name, _file_name)
  #   start_time = time.time()
  #   obj = self.optimizers.get_instance(request.client_id).get_weights(request)
  #   # calculate_resolve_time(start_time)
  #   logging.info("<<<<<< Out %s method of %s file.", method_name, _file_name)
  #   return obj

  def DecisionPoint(self, request, context):
    method_name = "DecisionPoint"
    logging.info(">>>>>>>  In %s method of %s file.", method_name, _file_name)
    start_time = time.time()
    obj = self.optimizers.get_instance(request.client_id).decision_point(
        request
    )
    # calculate_resolve_time(start_time)
    logging.info("<<<<<<<  Out %s method of %s file.", method_name, _file_name)
    return obj

  def CurrentStatus(self, request, context):
    method_name = "CurrentStatus"
    # logging.info(">>>>>>>  In %s method of %s file.", method_name, _file_name)

    return self.optimizers.get_instance(request.client_id).current_status(
        request
    )
    logging.info("<<<<<<<  Out %s method of %s file.", method_name, _file_name)

  def FetchOptimalAction(self, request, context):
    method_name = "FetchOptimalAction"
    logging.info(">>>>>>>  In %s method of %s file.", method_name, _file_name)

    obj = self.optimizers.get_instance(request.client_id).fetch_optimal_action(
        request
    )
    logging.info("<<<<<<<  Out %s method of %s file.", method_name, _file_name)
    return obj

  def ProposeAction(self, request, context):
    method_name = "ProposeAction"
    logging.info(">>>>>>>  In %s method of %s file.", method_name, _file_name)

    return self.optimizers.get_instance(request.client_id).propose_action(
        request
    )
    logging.info("<<<<<<<  Out %s method of %s file.", method_name, _file_name)

  def FinalizeEpisode(self, request, context):
    method_name = "FinalizeEpisode"
    logging.info(">>>>>>>  In %s method of %s file.", method_name, _file_name)

    obj = self.optimizers.get_instance(request.client_id).finalize_episode(
        request
    )
    logging.info("<<<<<<<  Out %s method of %s file.", method_name, _file_name)
    return obj

  def Launch(self, request, context):
    method_name = "Launch"
    logging.info(">>>>>>>  In %s method of %s file.", method_name, _file_name)
    # start_time = time.time()
    obj = self.optimizers.launch(request)
    # calculate_resolve_time(start_time)
    logging.info("<<<<<<<  Out %s method of %s file.", method_name, _file_name)
    return obj

  def Create(self, request, context):
    method_name = "Create"
    logging.info(">>>>>>>  In %s method of %s file.", method_name, _file_name)
    # start_time = time.time()
    unique_id = generate_unique_number()
    # calculate_resolve_time(start_time)

    logging.info("<<<<<<<  Out %s method of %s file.", method_name, _file_name)
    return service_pb2.CreateResponse(id=unique_id, path_prefix="/tmp/")

server = None
def serve():
  global server
  """Main method that listens on port 8080 and handle requests received from client.
  """
  method_name = "serve"
  logging.info(">>>>>>>  In %s method of %s file.", method_name, _file_name)

  server = grpc.server(
    futures.ThreadPoolExecutor(max_workers=500),
    options=[
        ('grpc.max_receive_message_length', 512 * 1024 * 1024),
    ]
  )
  service_pb2_grpc.add_SightServiceServicer_to_server(SightService(), server)
  server.add_insecure_port("[::]:9999")
  server.start()
  logging.info("server is up and running on port : 9999")

  flask_app.run(debug=True, host="0.0.0.0", port=8080)
  server.wait_for_termination()
  logging.info("<<<<<<<  Out %s method of %s file.", method_name, _file_name)

if __name__ == "__main__":
  method_name = "__main__"
  logging.basicConfig(level=logging.INFO)
  logging.info(">>>>>>>  In %s method of %s file.", method_name, _file_name)
  try:
    app.run(serve())
  except BaseException as e:
    logging.error("Error occurred : ")
    logging.error(e)
  logging.info("<<<<<<<  Out %s method of %s file.", method_name, _file_name)
