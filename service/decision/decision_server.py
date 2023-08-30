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

from concurrent import futures
import logging
import google.cloud.logging as log

import sys
import time
import grpc
from dotenv import load_dotenv
load_dotenv()
import os

from typing import Any
from datetime import datetime
from absl import flags
from google.cloud import aiplatform
from google.cloud import aiplatform_v1

from acme.agents.tf import dqn
from acme import specs
import dm_env
import numpy as np
import sonnet as snt


from service.decision import decision_resources
from service.decision import decision_pb2
from service.decision import decision_pb2_grpc

_VIZIER_TRAIN_EXPERIMENT_NAME = flags.DEFINE_string(
    'vizier_train_experiment_name', None,
    'The name of the Vizier experiment this worker will participate in.')

VIZIER_ENDPOINT = '{}-aiplatform.googleapis.com'.format('us-central1')
VIZIER_PARENT = 'projects/{}/locations/{}'.format('cameltrain', 'us-central1')
PROJECT_REGION = 'us-central1'
PROJECT_ID = 'cameltrain'
_vizier_client = aiplatform.gapic.VizierServiceClient(client_options=dict(api_endpoint=VIZIER_ENDPOINT))
            

_file_name = "decision_server.py"

instanceId = os.getenv('SPANNER_INSTANCE_ID')
databaseId = os.getenv('SPANNER_DATABASE_ID')
logtableId = os.getenv('SPANNER_LOG_TABLE_ID')
studytableId = os.getenv('SPANNER_STUDY_TABLE_ID')

action_dict_sorted = {}
state_dict_sorted = {}


def generateUniqueNumber():
    return int(time.time() * 1000)

def set_config_attributes(config_params, state_dict_sorted:dict, action_dict_sorted:dict):
    method_name = "set_config_attributes"
    # print(f">>>>>>>>>  In {method_name} of {_file_name}.")
    
    action_dict = {}
    for attr in config_params.action_attrs:
        action_dict[attr] = [config_params.action_attrs[attr].min_value, config_params.action_attrs[attr].max_value]
    # sorting dict key wise to maintain consistency at for all call
    action_keys = list(action_dict.keys())
    action_keys.sort()
    for k in action_keys : 
      action_dict_sorted[k] = action_dict[k]
    
    state_dict = {}
    for attr in config_params.state_attrs:
        state_dict[attr] = [config_params.state_attrs[attr].min_value, config_params.state_attrs[attr].max_value]
    # sorting dict key wise to maintain consistency at for all call
    state_keys = list(state_dict.keys())
    state_keys.sort()
    for k in state_keys : 
      state_dict_sorted[k] = state_dict[k]    
    # print(f"<<<<<<<<<  Out {method_name} of {_file_name}.")




def _get_vizier_study_display_name(id:str, label:str) -> str:
    return 'Sight_Decision_Study_' + label.replace(' ', '_') + '_' + str(
        id) + '_' + datetime.now().strftime('%Y%m%d_%H%M%S')

def _get_vizier_study_config(id:str, label:str, study_config_param):
  """Generate a Vizier StudyConfig from command-line flags."""
  study_params = []
  for attr in study_config_param.action_attrs:
    study_params.append({
    'parameter_id': attr,
    'double_value_spec': {
        'min_value': study_config_param.action_attrs[attr].min_value,
        'max_value': study_config_param.action_attrs[attr].max_value,
    }})
  return {
      'display_name': _get_vizier_study_display_name(id, label),
      'study_spec': {
        'algorithm': 'ALGORITHM_UNSPECIFIED',
        'parameters': study_params,
        'metrics': [{
            'metric_id': 'outcome',
            'goal': 'MAXIMIZE'
        }],
      }
  }

class SightService(decision_pb2_grpc.SightServiceServicer):

    def __init__(self):
        super().__init__()
        self.acme_agent = None
        self.selected_action = None
        # self.Trial = None
        self.obeserve_first = False
        self.current_state_values = None
        self.worker_trial_dict = {}
        self.client_dict = {}
        logging.info("updated client_dict = "+str(self.client_dict))

    def Test(self, request, context):
        print("executing the Test222")
        logging.info("executing the Test")
        obj = decision_pb2.TestResponse(val="222")
        print("obj is : ",obj)
        logging.info(obj)
        return obj

    def DecisionPointMethod(self, request, context):
        
        method_name = "DecisionPointMethod"
        logging.info(f">>>>>>>>>  In {method_name} of {_file_name} for worker {request.worker_id}.")
        # print("request here in DecisionPointMethod is : ", request)
        if request.worker_id not in self.client_dict[request.client_id].keys():
                self.client_dict[request.client_id][request.worker_id] = {}

        if(request.optimizer_type == decision_pb2.OptimizerType.OT_VIZIER):
            # print("Running for vizier flow....")
            response = _vizier_client.suggest_trials({
                            'parent': self.client_dict[request.client_id]["vizier_study"], #request.vizier_study_name, 
                            'suggestion_count': 1,
                            'client_id': request.worker_id,
                        }).result().trials

            self.client_dict[request.client_id][request.worker_id]["trial"] = response[0].name
            logging.info("updated client_dict = "+str(self.client_dict))
            
            obj = decision_pb2.DecisionPointResponse()
            for param in response[0].parameters:
                obj.actions[param.parameter_id] = param.value

        elif(request.optimizer_type == decision_pb2.OptimizerType.OT_ACME):
            # print("Running for acme flow....")
            
            # Extracting current state from request 
            state_val_list = []
            for state_val in request.decision_point.state_params.values():
                state_val_list.append(state_val)
            self.current_state_values = state_val_list
            # print("state_val_list : ",state_val_list)
            
            observation=np.array(state_val_list, dtype=np.float32)
            self.selected_action = self.acme_agent.select_action(observation)

            # we can get action keys from action_dict_sorted cached during launch
            received_action_dict = {}
            for key in action_dict_sorted.keys():
                #? need to change this if we have more than one action attr
                received_action_dict[key] = self.selected_action

            self.selected_action = np.array(self.selected_action, dtype=np.int32, ndmin=len(action_dict_sorted))  
            # print("self.selected_action : ", self.selected_action)

            # print("received_action_dict  :",received_action_dict)
            obj = decision_pb2.DecisionPointResponse(actions=received_action_dict)
        else:
            obj = decision_pb2.DecisionPointResponse(actions={request.optimizer_type+" - OPTIMIZER NOT VALID":0.0})
        
        # print("obj value is : ", obj)
        logging.info(f"<<<<<<<<<  Out {method_name} of {_file_name} for worker {request.worker_id}.")
        return obj

    def DecisionOutcomeMethod(self, request, context):
        method_name = "DecisionOutcomeMethod"
        logging.info(f">>>>>>>>>  In {method_name} of {_file_name} for worker {request.worker_id}.")
        # print("request here in DecisionOutcomeMethod : ", request)

        # This will be called only during last Decision Outcome call for vizier
        if(request.optimizer_type == decision_pb2.OptimizerType.OT_VIZIER):
            metrics = []
            metrics_obj = {}
            metrics_obj["metric_id"] = request.decision_outcome.outcome_label
            metrics_obj["value"] = request.decision_outcome.outcome_value
            metrics.append(metrics_obj)
            
            if request.worker_id in self.client_dict[request.client_id].keys() and "trial" in self.client_dict[request.client_id][request.worker_id].keys():
                _vizier_client.complete_trial({
                            'name': self.client_dict[request.client_id][request.worker_id]["trial"],
                            'final_measurement': {
                                'metrics': metrics
                                }
                        })
            else:
                logging.warn("Given worker not found......")
                logging.warn("current key(worker) is  = "+request.worker_id)
                logging.warn("current client_dict = "+str(self.client_dict))
            
            obj = decision_pb2.DecisionOutcomeResponse(response_str="Success!")
       
        # This will be called during each Decision Outcome call for Acme
        elif(request.optimizer_type == decision_pb2.OptimizerType.OT_ACME):
            # print("acme_agent : ", self.acme_agent)
            # if(self.obeserve_first == False):
            #     timestep = dm_env.TimeStep(
            #                 step_type=dm_env.StepType.FIRST,
            #                 reward=None,
            #                 discount=None,
            #                 observation=np.zeros(len(self.current_state_values), dtype=np.float32))
            #     self.acme_agent.observe_first(timestep)
            #     self.obeserve_first = True

            # If Decision outcome called from finalized episode, it will be last call 
            if request.last_call == True:
                # print("........Last Decision Outcome call........")
                current_step_type = dm_env.StepType.LAST
                # for next iteration, we have to call observe first
                self.obeserve_first = False
            else:
                current_step_type = dm_env.StepType.MID

            timestep = dm_env.TimeStep(
                        step_type=current_step_type,
                        reward=np.array(request.last_reward, dtype=np.float32),
                        discount=np.array(1., dtype=np.float32),
                        observation=np.array(self.current_state_values, dtype=np.float32))

            # print("self.selected_action in DO :", self.selected_action)
            # print("self.current_state_values in DO :", self.current_state_values)

            self.acme_agent.observe(self.selected_action, timestep)
            obj = decision_pb2.DecisionOutcomeResponse(response_str="Success!")
        else:
            obj = decision_pb2.DecisionOutcomeResponse(response_str="OPTIMIZER NOT VALID")
        
        # print("obj value is : ", obj)
        logging.info(f"<<<<<<<<<  Out {method_name} of {_file_name} for worker {request.worker_id}.")
        return obj

    def Launch(self, request, context):
        method_name = "Launch"
        logging.info(f">>>>>>>>>  In {method_name} of {_file_name}.")
        # print("request here in Launch : ", request)

        if(request.optimizer_type == decision_pb2.OptimizerType.OT_VIZIER):
            # print("Running for vizier flow....")
            study_config = _get_vizier_study_config(request.client_id, request.label, request.decision_config_params)                
            response = _vizier_client.create_study(parent=VIZIER_PARENT, study=study_config)
            vizier_URL = 'https://pantheon.corp.google.com/vertex-ai/locations/'+PROJECT_REGION+'/studies/'+response.name.split('/')[-1]+'?project='+PROJECT_ID

            self.client_dict[request.client_id]["vizier_study"] = response.name
            logging.info("updated client_dict = "+str(self.client_dict))
            obj = decision_pb2.LaunchResponse(display_string=vizier_URL)

        elif(request.optimizer_type == decision_pb2.OptimizerType.OT_ACME):
            # print("Running for acme flow....")

            # caching state, action attr with their min/max here
            state_dict_sorted.clear()
            action_dict_sorted.clear()
            set_config_attributes(request.decision_config_params, state_dict_sorted, action_dict_sorted)

            state_min_list = []
            state_max_list = []
            for val in state_dict_sorted.values():
                state_min_list.append(val[0])
                state_max_list.append(val[1])
            # print("state_min_list : ",state_min_list)
            # print("state_max_list : ",state_max_list)

            action_min_list = []
            action_max_list = []
            for val in action_dict_sorted.values():
                action_min_list.append(int(val[0]))
                action_max_list.append(int(val[1]))
            # print("action_min_list : ",action_min_list)
            # print("action_max_list : ",action_max_list)

            environment_spec = specs.EnvironmentSpec(
                    observations=specs.BoundedArray(shape=(len(state_dict_sorted),), dtype=np.float32, minimum=state_min_list, maximum=state_max_list, name='observation'),
                    actions=specs.BoundedArray(shape=(len(action_dict_sorted),), dtype=np.int32, minimum=action_min_list, maximum=action_max_list, name='action'),
                    rewards=specs.Array(shape=(), dtype=np.float32, name='reward'),
                    discounts= specs.BoundedArray(shape=(), dtype=np.float32, minimum=0., maximum=1., name='discount')
                )

            #? need to change this if we have more than one action attr
            policy_network = snt.Sequential([snt.Linear(action_max_list[0])])
            self.acme_agent  = dqn.DQN(environment_spec, policy_network)
            self.client_dict[request.id]["agent"] = self.acme_agent
            
            # observing with initial timestep
            timestep = dm_env.TimeStep(
                        step_type=dm_env.StepType.FIRST,
                        reward=None,
                        discount=None,
                        observation=np.zeros(len(self.current_state_values), dtype=np.float32))
            self.acme_agent.observe_first(timestep)

            # print("acme_agent : ", self.acme_agent)
            obj = decision_pb2.LaunchResponse(display_string="SUCCESS")
        else:
            obj = decision_pb2.LaunchResponse(display_string="OPTIMIZER NOT VALID!!")
        
        # print("obj value is : ", obj)
        logging.info(f"<<<<<<<<<  Out {method_name} of {_file_name}.")
        return obj

    def Create(self, request, context):
        method_name = "Create"
        logging.info(f">>>>>>>>>  In {method_name} of {_file_name}.")

        log_entry = {}
        # print("in create method of server.........")
        unique_id = generateUniqueNumber()
        log_entry["Id"] = unique_id
        self.client_dict[str(unique_id)] = {}
        logging.info("updated client_dict = "+str(self.client_dict))

        if(request.format == 0):
            logging.info("No log format specified. Defaulting to AVRO")
            request.format = 4
        log_entry["LogFormat"] = request.format

        if(not request.log_dir_path):
            logging.info("No LogPathPrefix specified, it will directly start from log...")
            final_path = "log_"+str(unique_id)
        else:
            final_path = request.log_dir_path+"log_"+str(unique_id)
        log_entry["LogPathPrefix"] = final_path

        if(not request.log_owner):
            logging.info("No log owner specified")
        else:
            log_entry["LogOwner"] = request.log_owner
    
        log_entry["LogLabel"] = request.label
        log_entry["Attribute"] = request.attribute

        # storing to GCP in spanner table
        decision_resources.Insert_In_LogDetails_Table(log_entry, instanceId, databaseId, logtableId)
        logging.info(f"<<<<<<<<<  Out {method_name} of {_file_name}.")
        return decision_pb2.CreateResponse(id=unique_id, path_prefix="/tmp/")

def serve():
    method_name = "serve"
    logging.info(f">>>>>>>>>  In {method_name} of {_file_name}.")

    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    decision_pb2_grpc.add_SightServiceServicer_to_server(SightService(), server)
    server.add_insecure_port('[::]:8080')
    server.start()
    logging.info("server is up and running on port : 8080")

    server.wait_for_termination()
    logging.info(f"<<<<<<<<<  Out {method_name} of {_file_name}.")
    
def createSpannerTable():
    decision_resources.create_database(instanceId, databaseId, logtableId, studytableId)

if __name__ == '__main__':
    # logging.basicConfig() #duplicating log on cloud run
    log_client = log.Client()
    log_client.setup_logging()
    
    method_name = "__main__"
    logging.info(f">>>>>>>>>  In {method_name} of {_file_name}.")

    try:
        createSpannerTable()
        serve()
    except BaseException as e:
        logging.error("Error occured")
        logging.error(e)
    
    logging.info(f"<<<<<<<<<  Out {method_name} of {_file_name}.")
