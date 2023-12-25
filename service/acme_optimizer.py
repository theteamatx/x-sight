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

"""Acme reinforcement learning for driving Sight applications."""

import concurrent.futures
import logging
from typing import Any, Dict, List, Optional, Tuple

import time
import acme
from acme import specs
from acme import wrappers
from acme.adders import reverb as adders_reverb
from acme.jax import utils
from acme.jax.experiments import config
import dm_env
import gym
import jax
import numpy as np
from overrides import overrides
from readerwriterlock import rwlock
import reverb
from service import service_pb2
# from service import server_utils
from service.build_learner import build_learner_config
from service.numproto.numproto import ndarray_to_proto
from service.numproto.numproto import proto_to_ndarray
from service.optimizer_instance import OptimizerInstance
from service.optimizer_instance import param_dict_to_proto
from service.optimizer_instance import param_proto_to_dict

_file_name = "acme_optimizer.py"


class Acme(OptimizerInstance):
  """Acme optimizer class to work with training methods.

  Attributes:
    agents: Maps each worker_id to the Agent object that learns from the
      worker's observations.
    last_action: Maps each worker_id to the most recent action taken by this
      worker in the current episode.
    _experiment: configurations of the problem
    _learner: learner object which will learn from the Trajectories
    _learner_weights_lock: Lock to transfer latest weights from learner
                            to _learner_weights
    _learner_weights: To maintain the latest weights till point from learner
    _replay_server: Replay buffer server where trajectories are stored
    _replay_client: Replay client to communicate with server
    _dataset: Iterator to fetch data from replay buffer server
    _learner_checkpointer:
  """

  def __init__(self):
    super().__init__()
    self._experiment = None
    self._learner = None
    self._learner_weights_lock = rwlock.RWLockFair()
    self._learner_weights = None
    self._replay_server = None
    self._replay_client = None
    self._dataset = None
    self._learner_checkpointer = None
    # self._last_updated_at = 0
    self._avg_insertion_time = []
    self._avg_updation_time = []

  def print_insertion_time(self):
    logging.info("all insertion times : %s", self._avg_insertion_time)

  def calculate_time(self, start_time, operation):
    method_name = "calculate_time"
    logging.debug(">>>>  In %s of %s", method_name, _file_name)

    current_time = time.time() - start_time
    if(operation == 'insert'):
      self._avg_insertion_time.append(round(current_time,2))
      avg_time = sum(self._avg_insertion_time) / len(self._avg_insertion_time)
    elif(operation == 'update'):
      self._avg_updation_time.append(current_time)
      avg_time = sum(self._avg_updation_time) / len(self._avg_updation_time)
    logging.info("%s Time: - latest time : %s and Average time : %s seconds",
                 operation, round(current_time, 3), round(avg_time, 3))
    logging.debug("<<<<  Out %s of %s", method_name, _file_name)



  def fetch_replay_table_size(self):
    method_name = "fetch_replay_table_size"
    logging.debug(">>>>  In %s of %s", method_name, _file_name)
    table_info = self._replay_client.server_info()
    table_size = table_info[
        adders_reverb.DEFAULT_PRIORITY_TABLE
    ].rate_limiter_info.insert_stats.completed
    logging.debug("<<<<  Out %s of %s", method_name, _file_name)
    return table_size

  def update_learner(self):
    method_name = "update_learner"
    logging.debug(">>>>  In %s of %s", method_name, _file_name)
    try:
      while True:
        # If dataset iterator has enough data to sample
        if self._dataset.ready():
          logging.info("updating learner................")
          start_time = time.time()
          self._learner.step()
          # self.calculate_time(start_time, 'update')

          # transfering updated learner weights to _learner_weights
          # variable with write lock
          with self._learner_weights_lock.gen_wlock():
            self._learner_weights = self._learner.get_variables("")
      logging.debug("<<<<  Out %s of %s", method_name, _file_name)
    except Exception as e:
      logging.exception("Exception in learner thread : %s", e)

  def insert_to_replay(self, request):
    method_name = "insert_to_replay"
    logging.debug(">>>>  In %s of %s", method_name, _file_name)
    try:
      if request.acme_decision_point.episode_observations:
        # logging.info(
        #     "adding this many data into buffer via thread : %d",
        #     len(request.acme_decision_point.episode_observations),
        # )

        start_time = time.time()
        adder = self._experiment.builder.make_adder(
            self._replay_client, self._environment_spec, self._policy
        )

        episode_observations = request.acme_decision_point.episode_observations
        for episode_obs in episode_observations:
          if episode_obs.HasField("action"):
            action = proto_to_ndarray(episode_obs.action)
          else:
            action = np.array(0, dtype=np.int64)
          if episode_obs.HasField("reward"):
            reward = proto_to_ndarray(episode_obs.reward)
          if episode_obs.HasField("discount"):
            discount = proto_to_ndarray(episode_obs.discount)
          else:
            discount = np.array(0, dtype=np.float64)
          observation = proto_to_ndarray(episode_obs.observation)
          steptype = episode_obs.steptype

          if steptype == dm_env.StepType.FIRST:
            action = None
            timestep = dm_env.TimeStep(
                step_type=steptype,
                reward=None,
                discount=None,
                observation=observation,
            )
            adder.add_first(timestep)
          else:
            timestep = dm_env.TimeStep(
                step_type=steptype,
                reward=reward,
                discount=discount,
                observation=observation,
            )
            adder.add(action, timestep)
        # self.calculate_time(start_time, 'insert')
        logging.info("table_size: %s", self.fetch_replay_table_size())
      else:
        logging.info("no data to insert.....")
      logging.debug("<<<<  Out %s of %s", method_name, _file_name)
    except Exception as e:
      logging.exception("Exception in thread : %s", e)

  @overrides
  def get_weights(
      self, request: service_pb2.GetWeightsRequest
  ) -> service_pb2.GetWeightsResponse:
    method_name = "get_weights"
    logging.debug(">>>>  In %s of %s", method_name, _file_name)

    executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    executor.submit(self.insert_to_replay, request)
    # Manually shutdown the executor after submitting tasks
    executor.shutdown(wait=False)

    # logging.info("sending latest weights back to client")
    with self._learner_weights_lock.gen_rlock():
      latest_weights = self._learner_weights

    weights_msg = service_pb2.GetWeightsResponse()
    for layer_data in latest_weights:
      for layer_name, layer_info in layer_data.items():
        layer_msg = weights_msg.layers.add()
        layer_msg.name = layer_name

        weights_data_msg = layer_msg.weights
        weights_data_msg.b.extend(layer_info["b"])
        weights_data_msg.w.CopyFrom(ndarray_to_proto(layer_info["w"]))
    # print(f"<<<<  Out {method_name} of {_file_name}")
    logging.debug("<<<<  Out %s of %s", method_name, _file_name)
    return weights_msg

  def generate_env_spec(self, acme_config):
    method_name = "generate_env_spec"
    logging.debug(">>>>  In %s of %s", method_name, _file_name)
    state_min = np.array(list(acme_config.state_min))
    state_max = np.array(list(acme_config.state_max))
    state_param_length = acme_config.state_param_length
    action_min = np.array(list(acme_config.action_min))
    action_max = np.array(list(acme_config.action_max))
    action_param_length = acme_config.action_param_length

    new_env_spec = specs.EnvironmentSpec(
      observations=specs.BoundedArray(shape=(state_param_length,), dtype=np.float32, name='observation', minimum=state_min, maximum=state_max),
      # actions=specs.DiscreteArray(dtype=np.int64, name='action', num_values=possible_actions),
      #? works for only 1 action parameter
      actions=specs.BoundedArray(shape=(), dtype=int, name='action', minimum=action_min[0], maximum=action_max[0]),
      rewards=specs.Array(shape=(), dtype=np.float64, name='reward'),
      discounts=specs.BoundedArray(shape=(), dtype=np.float64, name='discount', minimum=0.0, maximum=1.0)
    )
    logging.debug("<<<<  Out %s of %s", method_name, _file_name)
    return new_env_spec

  def create_learner(self, client_id, acme_config):
    method_name = "create_learner"
    logging.debug(">>>>  In %s of %s", method_name, _file_name)

    if(acme_config.env_name):
      print("env found, using that to generate spec")
      self._experiment = build_learner_config(env_name=acme_config.env_name)
      environment = self._experiment.environment_factory()
      print('environment : ', environment)
      environment_spec = specs.make_environment_spec(environment)
    else:
      print("env not found, directly generating spec")
      self._experiment = build_learner_config(possible_action_values=acme_config.possible_actions)
      environment_spec = self.generate_env_spec(acme_config)

    networks = self._experiment.network_factory(environment_spec)
    policy = config.make_policy(
        experiment=self._experiment,
        networks=networks,
        environment_spec=environment_spec,
        evaluation=False,
    )
    replay_tables = self._experiment.builder.make_replay_tables(
        environment_spec, policy
    )

    replay_server = reverb.Server(replay_tables, port=None)
    replay_client = reverb.Client(f"localhost:{replay_server.port}")

    dataset = self._experiment.builder.make_dataset_iterator(replay_client)
    dataset = utils.prefetch(dataset, buffer_size=1)

    key = jax.random.PRNGKey(0)
    learner_key, key = jax.random.split(key)
    learner = self._experiment.builder.make_learner(
        random_key=learner_key,
        networks=networks,
        dataset=dataset,
        logger_fn=self._experiment.logger_factory,
        environment_spec=environment_spec,
        replay_client=replay_client,
    )

    self._learner = learner
    self._replay_server = replay_server
    self._replay_client = replay_client
    self._dataset = dataset
    self._environment_spec = environment_spec
    self._policy = policy

    with self._learner_weights_lock.gen_wlock():
      self._learner_weights = self._learner.get_variables("")

    # spinning a thread which update the learner when dataset iterator is ready
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    executor.submit(self.update_learner)
    # Manually shutdown the executor after submitting tasks
    executor.shutdown(wait=False)
    logging.debug("<<<<  Out %s of %s", method_name, _file_name)
    # print(f"<<<<  Out {method_name} of {_file_name}")

  @overrides
  def launch(
      self, request: service_pb2.LaunchRequest
  ) -> service_pb2.LaunchResponse:
    method_name = "launch"
    logging.debug(">>>>  In %s of %s", method_name, _file_name)
    super(Acme, self).launch(request)

    # self.create_learner(request.client_id, request.acme_config.env_name)
    
    self.create_learner(request.client_id, 
                        llm_config = request.decision_config_params.choice_config[request.label].acme_config)

    # TODO : meetashah : this is old version, might have to modify for now.
    # storing client details in case server crashed - mid run
    # client_details = {}
    # client_details["sight_id"] = int(request.client_id)
    # client_details["env"] = "CartPole-v1"
    # client_details["network_path"] = (
    #     f"gs://{FLAGS.project_id}-sight/learner/" + request.client_id + "/"
    # )
    # client_details["learner_path"] = (
    #     f"gs://{FLAGS.project_id}-sight/learner/" + request.client_id + "/"
    # )
    # client_details["replay_address"] = "127.0.0.1"
    # server_utils.Insert_In_ClientData_Table(
    #     client_details, "sight-data", "sight_db", "ClientData"
    # )

    response = service_pb2.LaunchResponse()
    response.display_string = "ACME SUCCESS!"
    logging.debug("<<<<  Out %s of %s", method_name, _file_name)
    return response

  @overrides
  def decision_point(
      self, request: service_pb2.DecisionPointRequest
  ) -> service_pb2.DecisionPointResponse:
    method_name = "decision_point"
    logging.debug(">>>>  In %s of %s", method_name, _file_name)
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    executor.submit(self.insert_to_replay, request)
    # Manually shutdown the executor after submitting tasks
    executor.shutdown(wait=False)

    # logging.info("sending latest weights back to client")
    with self._learner_weights_lock.gen_rlock():
      latest_weights = self._learner_weights

    response = service_pb2.DecisionPointResponse()
    weights_msg = service_pb2.Acme_Response()
    for layer_data in latest_weights:
      for layer_name, layer_info in layer_data.items():
        layer_msg = weights_msg.layers.add()
        layer_msg.name = layer_name

        weights_data_msg = layer_msg.weights
        weights_data_msg.b.extend(layer_info["b"])
        weights_data_msg.w.CopyFrom(ndarray_to_proto(layer_info["w"]))
    response.acme_response.CopyFrom(weights_msg)
    # print(f"<<<<  Out {method_name} of {_file_name}")
    logging.debug("<<<<  Out %s of %s", method_name, _file_name)
    return response

  @overrides
  def finalize_episode(
      self, request: service_pb2.FinalizeEpisodeRequest
  ) -> service_pb2.FinalizeEpisodeResponse:
    method_name = "finalize_episode"
    logging.debug(">>>>  In %s of %s", method_name, _file_name)

    observation = np.array(
        list(param_proto_to_dict(request.decision_point.state_params).values()),
        dtype=np.float32,
    )
    # logging.info('observation : %s', observation)
    with self.last_action_lock.gen_wlock():
      if request.worker_id in self.last_action:
        action = self.last_action[request.worker_id]

        timestep = dm_env.TimeStep(
            step_type=dm_env.StepType.LAST,
            reward=np.array(
                request.decision_outcome.outcome_value, dtype=np.float64
            ),
            discount=np.array(
                request.decision_outcome.discount, dtype=np.float64
            ),
            observation=np.frombuffer(observation, dtype=np.float32),
        )

        with self.agents_lock.gen_rlock():
          self.agents[request.worker_id].observe(
              np.int64(action), next_timestep=timestep
          )
          self.agents[request.worker_id].update()
        self._learner_checkpointer.save(force=True)

        # Resetting last action for agent since it is the end of the episode.
        del self.last_action[request.worker_id]

    logging.debug("<<<<  Out %s of %s", method_name, _file_name)
    return service_pb2.FinalizeEpisodeResponse(response_str="Success!")

  @overrides
  def current_status(
      self, request: service_pb2.CurrentStatusRequest
  ) -> service_pb2.CurrentStatusResponse:
    method_name = "current_status"
    logging.debug(">>>>  In %s of %s", method_name, _file_name)
    response = "[ACME]\n"
    logging.debug("<<<<  Out %s of %s", method_name, _file_name)
    return service_pb2.CurrentStatusResponse(response_str=response)
