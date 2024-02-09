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
import time
import json
from acme import specs
from acme.adders import reverb as adders_reverb
from acme.jax import utils
from acme.jax.experiments import config
import dm_env
import jax
import numpy as np
from overrides import overrides
from readerwriterlock import rwlock
import reverb
from sight.proto import sight_pb2
from sight_service.proto import service_pb2
# from service import server_utils
from sight_service.build_dqn_learner import build_dqn_config
from sight_service.build_d4pg_learner import build_d4pg_config
from sight_service.proto.numproto.numproto import ndarray_to_proto
from sight_service.proto.numproto.numproto import proto_to_ndarray
from sight_service.optimizer_instance import OptimizerInstance
from sight_service.optimizer_instance import param_dict_to_proto

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
    self._learner_weights = {}
    self._replay_server = None
    self._replay_client = None
    self._dataset = None
    self._learner_checkpointer = None
    self._learner_keys = ["policy", "critic"]
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
    logging.info(">>>>  In %s of %s", method_name, _file_name)
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
            all_weights = self._learner.get_variables(self._learner_keys)
            for i in range(len(all_weights)):
              self._learner_weights[self._learner_keys[i]] = all_weights[i]
      logging.info("<<<<  Out %s of %s", method_name, _file_name)
    except Exception as e:
      logging.exception("Exception in learner thread : %s", e)

  def insert_to_replay(self, request):
    method_name = "insert_to_replay"
    logging.info(">>>>  In %s of %s", method_name, _file_name)
    try:
      if request.acme_config.episode_observations:
        # logging.info(
        #     "adding this many data into buffer via thread : %d",
        #     len(request.acme_config.episode_observations),
        # )

        start_time = time.time()
        adder = self._experiment.builder.make_adder(
            self._replay_client, self._environment_spec, self._policy
        )

        episode_observations = request.acme_config.episode_observations
        for episode_obs in episode_observations:
          if episode_obs.HasField("action"):
            action = proto_to_ndarray(episode_obs.action)
          else:
            action = np.array(0, dtype=np.int64)
            # todo : meetashah - changed dtyep from int64 to float64 for d4pg agent
            # action = np.array(0, dtype=np.float32)
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
            # print("first timestep : ", timestep)
            # raise SystemExit
            adder.add_first(timestep)
          else:
            timestep = dm_env.TimeStep(
                step_type=steptype,
                reward=reward,
                discount=discount,
                observation=observation,
            )
            # print("mid timestep : ", timestep)
            # print("action : ", action, type(action), action.shape)
            # raise SystemExit
            adder.add(action, timestep)
        # self.calculate_time(start_time, 'insert')
        logging.info("table_size: %s", self.fetch_replay_table_size())
      else:
        logging.info("no data to insert.....")
      logging.debug("<<<<  Out %s of %s", method_name, _file_name)
    except Exception as e:
      logging.exception("Exception in thread : %s", e)

  # @overrides
  # def get_weights(
  #     self, request: service_pb2.GetWeightsRequest
  # ) -> service_pb2.GetWeightsResponse:
  #   method_name = "get_weights"
  #   logging.debug(">>>>  In %s of %s", method_name, _file_name)

  #   executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
  #   executor.submit(self.insert_to_replay, request)
  #   # Manually shutdown the executor after submitting tasks
  #   executor.shutdown(wait=False)

  #   # logging.info("sending latest weights back to client")
  #   with self._learner_weights_lock.gen_rlock():
  #     latest_weights = self._learner_weights

  #   weights_msg = service_pb2.GetWeightsResponse()
  #   for layer_data in latest_weights:
  #     for layer_name, layer_info in layer_data.items():
  #       layer_msg = weights_msg.layers.add()
  #       layer_msg.name = layer_name


  #       weights_data_msg = layer_msg.weights
  #       if "offset" in layer_info:
  #         weights_data_msg.offset.extend(layer_info["offset"])
  #       if "scale" in layer_info:
  #         weights_data_msg.scale.extend(layer_info["scale"])
  #       if "b" in layer_info:
  #         weights_data_msg.b.extend(layer_info["b"])
  #       if "w" in layer_info:
  #         weights_data_msg.w.CopyFrom(ndarray_to_proto(layer_info["w"]))
  #   # print(f"<<<<  Out {method_name} of {_file_name}")
  #   logging.debug("<<<<  Out %s of %s", method_name, _file_name)
  #   return weights_msg

  def generate_env_spec(self, state_attrs, action_attrs):
    method_name = "generate_env_spec"
    logging.debug(">>>>  In %s of %s", method_name, _file_name)
    # state_min = np.array(list(acme_config.state_min))
    # state_max = np.array(list(acme_config.state_max))
    # state_param_length = acme_config.state_param_length
    # action_min = np.array(list(acme_config.action_min))
    # action_max = np.array(list(acme_config.action_max))
    # action_param_length = acme_config.action_param_length

    # new_env_spec = specs.EnvironmentSpec(
    #   observations=specs.BoundedArray(shape=(state_param_length,), dtype=np.float32, name='observation', minimum=state_min, maximum=state_max),
    #   # actions=specs.DiscreteArray(dtype=np.int64, name='action', num_values=possible_actions),
    #   #? works for only 1 action parameter
    #   actions=specs.BoundedArray(shape=(), dtype=int, name='action', minimum=action_min[0], maximum=action_max[0]),
    #   rewards=specs.Array(shape=(), dtype=np.float64, name='reward'),
    #   discounts=specs.BoundedArray(shape=(), dtype=np.float64, name='discount', minimum=0.0, maximum=1.0)
    # )

    state_min = []
    state_max = []
    action_min = []
    action_max = []

    state_dtype = None
    action_dtype = None

    dtype_mapping = {
        sight_pb2.DecisionConfigurationStart.DataType.DT_INT32: np.int32,
        sight_pb2.DecisionConfigurationStart.DataType.DT_INT64: np.int64,
        sight_pb2.DecisionConfigurationStart.DataType.DT_FLOAT32: np.float32,
        sight_pb2.DecisionConfigurationStart.DataType.DT_FLOAT64: np.float64,
    }

    for key, attr_props in state_attrs.items():
      state_min.append(attr_props.min_value)
      state_max.append(attr_props.max_value)
      if(state_dtype == None):
        state_dtype = dtype_mapping[attr_props.datatype]
    observations = specs.BoundedArray(shape=(len(state_max),), dtype=state_dtype, name='observation', minimum=state_min, maximum=state_max)

    valid_action_values = None
    for key, attr_props in action_attrs.items():
      action_min.append(attr_props.min_value)
      action_max.append(attr_props.max_value)
      if(action_dtype == None):
        action_dtype = dtype_mapping[attr_props.datatype]
      if(attr_props.valid_int_values):
        valid_action_values = attr_props.valid_int_values

    # print('action_attrs : ', action_attrs)

    if(valid_action_values):
      actions = specs.DiscreteArray(num_values=valid_action_values,dtype=action_dtype,name="action")
    else:
      actions = specs.BoundedArray(shape=(len(action_max),), dtype=action_dtype, name='action', minimum=action_min, maximum=action_max)

    # print(state_min, state_max, len(state_max), state_dtype)
    # print(action_min, action_max, len(action_max), action_dtype)

    new_env_spec = specs.EnvironmentSpec(
      observations=observations,
      actions=actions,
      rewards=specs.Array(shape=(), dtype=np.float64, name='reward'),
      discounts=specs.BoundedArray(shape=(), dtype=np.float64, name='discount', minimum=0.0, maximum=1.0)
    )
    # print('new_env_spec : ', new_env_spec)

    logging.debug("<<<<  Out %s of %s", method_name, _file_name)
    return new_env_spec

  def create_learner(self, client_id, acme_config, state_attrs, action_attrs):
    method_name = "create_learner"
    logging.info(">>>>  In %s of %s", method_name, _file_name)

    if(False):
    # if(True):
      # if acme_config.env_name == "Pendulum-v1":
      #   self._experiment = build_d4pg_config(env_name=acme_config.env_name)
      # elif acme_config.env_name == "CartPole-v1":
      self._experiment = build_dqn_config(env_name="CartPole-v1")
      environment = self._experiment.environment_factory()
      environment_spec = specs.make_environment_spec(environment)
      print('If environment_spec : ', environment_spec)
    else:
      if(acme_config.acme_agent == sight_pb2.DecisionConfigurationStart.AcmeConfig.AA_DQN):
        # for action, attributes in action_attrs.items():
        #   possible_actions=attributes.valid_int_values
        # print('possible_actions : ', possible_actions)
        # self._experiment = build_dqn_config(possible_action_values=possible_actions)
        self._experiment = build_dqn_config()
      elif(acme_config.acme_agent == sight_pb2.DecisionConfigurationStart.AcmeConfig.AA_D4PG):
        self._experiment = build_d4pg_config()

      environment_spec = self.generate_env_spec(state_attrs, action_attrs)

      # print('Else environment_spec : ', environment_spec)


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
    print("learner : ", learner)


    self._learner = learner
    self._replay_server = replay_server
    self._replay_client = replay_client
    self._dataset = dataset
    self._environment_spec = environment_spec
    self._policy = policy

    # keeping weights in learner_weights dict so, future calls can directly get the 'at time updated weights'
    with self._learner_weights_lock.gen_wlock():
      all_weights = self._learner.get_variables(self._learner_keys)
      for i in range(len(all_weights)):
        self._learner_weights[self._learner_keys[i]] = all_weights[i]

    # spinning a thread which update the learner when dataset iterator is ready
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    executor.submit(self.update_learner)
    # Manually shutdown the executor after submitting tasks
    executor.shutdown(wait=False)
    logging.info("<<<<  Out %s of %s", method_name, _file_name)
    # print(f"<<<<  Out {method_name} of {_file_name}")

  @overrides
  def launch(
      self, request: service_pb2.LaunchRequest
  ) -> service_pb2.LaunchResponse:
    method_name = "launch"
    logging.debug(">>>>  In %s of %s", method_name, _file_name)
    super(Acme, self).launch(request)

    print('launch request : ', request)


    # self.create_learner(request.client_id, request.acme_config.env_name)
    # self.create_learner(request.client_id, request.acme_config)

    self.create_learner(request.client_id,
                        acme_config = request.decision_config_params.choice_config[request.label].acme_config,
                        state_attrs = request.decision_config_params.state_attrs,
                        action_attrs = request.decision_config_params.action_attrs
                      )

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
    logging.info("<<<<  Out %s of %s", method_name, _file_name)
    return response

  @overrides
  def decision_point(
      self, request: service_pb2.DecisionPointRequest
  ) -> service_pb2.DecisionPointResponse:
    method_name = "decision_point"
    logging.info(">>>>  In %s of %s", method_name, _file_name)

    #? start separate thread for the insertion to replay
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    executor.submit(self.insert_to_replay, request)
    # Manually shutdown the executor after submitting tasks
    executor.shutdown(wait=False)

    # logging.info("sending latest weights back to client")
    latest_weights = []
    with self._learner_weights_lock.gen_rlock():
      if(len(request.acme_config.learner_keys)>0):
        for key in request.acme_config.learner_keys:
          latest_weights.append(self._learner_weights[key])
      else:
        latest_weights.append(self._learner_weights["policy"])

    response = service_pb2.DecisionPointResponse()

    # Convert NumPy arrays to lists before serialization
    def convert_np_to_list(obj):
      if isinstance(obj, np.ndarray):
          return {'data': obj.tolist(), 'shape': obj.shape}
      return obj

    # directly serializing the weights structure
    serialized_weights = json.dumps(latest_weights, default=convert_np_to_list).encode('utf-8')
    response.weights = serialized_weights

    logging.info("<<<<  Out %s of %s", method_name, _file_name)
    return response

  @overrides
  def finalize_episode(
      self, request: service_pb2.FinalizeEpisodeRequest
  ) -> service_pb2.FinalizeEpisodeResponse:
    method_name = "finalize_episode"
    logging.debug(">>>>  In %s of %s", method_name, _file_name)

    executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    executor.submit(self.insert_to_replay, request)
    # Manually shutdown the executor after submitting tasks
    executor.shutdown(wait=False)

    # observation = np.array(
    #     list(param_proto_to_dict(request.decision_point.state_params).values()),
    #     dtype=np.float32,
    # )
    # # logging.info('observation : %s', observation)
    # with self.last_action_lock.gen_wlock():
    #   if request.worker_id in self.last_action:
    #     action = self.last_action[request.worker_id]

    #     timestep = dm_env.TimeStep(
    #         step_type=dm_env.StepType.LAST,
    #         reward=np.array(
    #             request.decision_outcome.outcome_value, dtype=np.float64
    #         ),
    #         discount=np.array(
    #             request.decision_outcome.discount, dtype=np.float64
    #         ),
    #         observation=np.frombuffer(observation, dtype=np.float32),
    #     )

    #     with self.agents_lock.gen_rlock():
    #       self.agents[request.worker_id].observe(
    #           np.int64(action), next_timestep=timestep
    #       )
    #
    #       # self.agents[request.worker_id].observe(
    #       #     np.float32(action), next_timestep=timestep
    #       # )
    #       self.agents[request.worker_id].update()
        # self._learner_checkpointer.save(force=True)

        # Resetting last action for agent since it is the end of the episode.
        # del self.last_action[request.worker_id]

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
