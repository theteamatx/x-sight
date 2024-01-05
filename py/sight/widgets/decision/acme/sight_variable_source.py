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

  def postprocess_data(self, latest_weights):
    result_dict = {}
    for record in latest_weights:
      name = record["name"]
      weights = record["weights"]
      result_dict[name] = weights
    return result_dict

  def proto_to_weights(self, proto_weights):
    method_name = "observation_to_proto"
    logging.debug(">>>>  In %s of %s", method_name, _file_name)

    updated_weights = []
    for layer in proto_weights.layers:
      layer_dict = {
          "name": layer.name,
          "weights": {
              "b": jnp.array(layer.weights.b),
              "w": jnp.array(proto_to_ndarray(layer.weights.w)),
          },
      }
      updated_weights.append(layer_dict)
    logging.debug("<<<<  Out %s of %s", method_name, _file_name)
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
    request = self._adder.fetch_and_reset_observation_list(
        self._client_id, self._worker_id
    )

    start_time = time.time()
    response = service.call(
        lambda s, meta: s.DecisionPoint(request, 300, metadata=meta)
    )
    # self.calculate_response_time(start_time)

    learner_weights = response.acme_response

    latest_weights = self.proto_to_weights(learner_weights)
    weights = self.postprocess_data(latest_weights)
    # print('weights : ', weights)
    logging.debug("<<<<  Out %s of %s", method_name, _file_name)
    return weights
