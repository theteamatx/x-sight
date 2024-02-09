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

"""Custom implementation of core variable_source."""
import os
import time
import numpy as np
import json
from typing import Any, List, Sequence
from absl import flags, logging
from acme import core
from acme import types
import jax.numpy as jnp
from sight_service.proto import service_pb2
from sight_service.proto.numproto.numproto import ndarray_to_proto, proto_to_ndarray
from sight import data_structures
from sight import service_utils as service

from sight.widgets.decision.acme import sight_adder

_file_name = "custom_variable_source.py"


# Convert lists back to NumPy arrays during deserialization
def convert_list_to_np(obj):
  if 'data' in obj and 'shape' in obj:
      return np.array(obj['data']).reshape(obj['shape'])
  return obj


class SightVariableSource(core.VariableSource):
  """A custom variable_source based on the core with some logic changes.

  This variable source implementation calls sight server to fetch the latest
  weights when get_variables method is called and pass the list from custom
  adder to send observation data gathered till point.
  """

  def __init__(self, client_id, adder: sight_adder.SightAdder, sight):
    self._adder = adder
    self._client_id = client_id
    self._worker_id = None
    self._response_time = []
    self._sight = sight

  def postprocess_data(self, networks_weights):
    result_dict = {}
    for record in latest_weights:
      name = record["name"]
      weights = record["weights"]
      result_dict[name] = weights
    return result_dict

  def proto_to_weights(self, networks_weights):
    method_name = "observation_to_proto"
    logging.debug(">>>>  In %s of %s", method_name, _file_name)
    # print("networks_weights : ", networks_weights)
    # raise SystemExit

    updated_weights = []
    # for layer in proto_weights.layers:
    for network in networks_weights.acme_response:
      # to store inidividual network weights (policy/critic)
      network_weight = []
      for layer in network.layers:
        layer_dict = {}
        layer_dict['name'] = layer.name
        # print("layer.name : ", layer.name)
        layer_dict['weights'] = {}
        if layer.weights.offset:
          layer_dict['weights']['offset'] = jnp.array(layer.weights.offset)
        if layer.weights.scale:
          layer_dict['weights']['scale'] = jnp.array(layer.weights.scale)
        if layer.weights.b:
          layer_dict['weights']['b'] = jnp.array(layer.weights.b)
        if layer.weights.w and len(layer.weights.w.ndarray) > 0:
          layer_dict['weights']['w'] = jnp.array(proto_to_ndarray(layer.weights.w))
      # layer_dict = {
      #     "name": layer.name,
      #     "weights": {
      #         "b": jnp.array(layer.weights.b),
      #         "w": jnp.array(proto_to_ndarray(layer.weights.w)),
      #     },
      # }
        network_weight.append(layer_dict)
      # print('network_weight : ', network_weight)
      updated_weights.append(network_weight)
    logging.debug("<<<<  Out %s of %s", method_name, _file_name)
    print('updated_weights : ', updated_weights)
    # raise SystemExit
    return updated_weights

  # to measure the latency call
  def calculate_response_time(self, start_time):
    response_time = time.time() - start_time
    data_structures.log(round(response_time, 4), self._sight)
    print(f"Response Time From Server: {round(response_time,4)} seconds")

  def get_variables(self, names: Sequence[str]) -> List[types.NestedArray]:
    method_name = "get_variables"
    logging.debug(">>>>  In %s of %s", method_name, _file_name)

    if flags.FLAGS.deployment_mode == "local" or flags.FLAGS.trained_model_log_id:
      self._worker_id = 0
    else:
      self._worker_id = os.environ["worker_location"]

    # if len(self._adder._observation_list) > 0:
    request, final_observation = self._adder.fetch_and_reset_observation_list(
        self._client_id, self._worker_id, names
    )
    # print("request here is : ", request)
    # raise SystemExit

    start_time = time.time()
    if final_observation:
      response = service.call(
          lambda s, meta: s.FinalizeEpisode(request, 300, metadata=meta)
      )
    else:
      response = service.call(
          lambda s, meta: s.DecisionPoint(request, 300, metadata=meta)
      )
    # self.calculate_response_time(start_time)

    # print("response : ", response)
    # learner_weights = response.acme_response
    # print("learner_weights : ", learner_weights)
    # raise SystemExit

    # latest_weights = self.proto_to_weights(response)

    weights = json.loads(response.weights.decode('utf-8'), object_hook=convert_list_to_np)
    # weights = self.postprocess_data(latest_weights)
    # it returns weights as list
    # print('weights : ', weights)
    # raise SystemExit
    # logging.info("<<<<  Out %s of %s", method_name, _file_name)
    return weights[0]
