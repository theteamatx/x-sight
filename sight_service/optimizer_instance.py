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

"""An instance of a Sight optimizer dedicated to a single experiment."""

from concurrent import futures
import logging
from typing import Any, Dict, List, Tuple, Sequence
from sight.widgets.decision import utils
from sight_service.proto import service_pb2
from sight.proto import sight_pb2

_file_name = "optimizer_instance.py"

def param_dict_to_proto(
    param_dict: Dict[str, float]
) -> List[sight_pb2.DecisionParam]:
  """converting dictionary of parameters into proto."""
  param_proto: List[sight_pb2.DecisionParam] = []
  for k, v in sorted(param_dict.items()):
    if isinstance(v, str):
      val = sight_pb2.Value(
                  sub_type=sight_pb2.Value.ST_STRING,
                  string_value=v,
              )
    elif isinstance(v, float):
      val = sight_pb2.Value(
                  sub_type=sight_pb2.Value.ST_DOUBLE,
                  double_value=v,
              )
    elif (not utils.is_scalar(v)):
      print('here v is : ', v, type(v))
      val = sight_pb2.Value(
                  sub_type=sight_pb2.Value.ST_JSON,
                  json_value=v,
              )
    else:
      raise ValueError('action attribute type must be either string or float')

    param_proto.append(
        sight_pb2.DecisionParam(
            key=k,
            value=val
        )
    )
  return param_proto


def param_proto_to_dict(
    param_proto: Sequence[sight_pb2.DecisionParam],
) -> Dict[str, float]:
  """converting proto back into dictionary of parameters."""
  param_dict = {}
  for param in param_proto:
    # if ((param.value.sub_type != sight_pb2.Value.ST_DOUBLE) and (param.value.sub_type != sight_pb2.Value.ST_STRING)):
    #   raise ValueError("Unsupported action type %s" % param.value.sub_type)
    # param_dict[param.key] = param.value.double_value
    if (param.value.sub_type == sight_pb2.Value.ST_DOUBLE):
      param_dict[param.key] = param.value.double_value
    elif (param.value.sub_type == sight_pb2.Value.ST_STRING):
      param_dict[param.key] = param.value.string_value
    elif (param.value.sub_type == sight_pb2.Value.ST_BOOL):
      param_dict[param.key] = param.value.bool_value
    elif (param.value.sub_type == sight_pb2.Value.ST_BYTES):
      param_dict[param.key] = param.value.bytes_value
    elif (param.value.sub_type == sight_pb2.Value.ST_INT64):
      param_dict[param.key] = param.value.int64_value
    elif (param.value.sub_type == sight_pb2.Value.ST_JSON):
      param_dict[param.key] = param.value.json_value
    else:
      raise ValueError("Unsupported action type %s" % param.value.sub_type)
  return param_dict


class OptimizerInstance:
  """An OptimizerInstance class that is generic for all optimizers.

  An optimizer containing base methods which specialized optimizers will
  override while communicating with client.
  """

  def __init__(self):
    self.actions = {}
    self.state = {}
    self.outcomes = {}

  def launch(
      self, request: service_pb2.LaunchRequest
  ) -> service_pb2.LaunchResponse:
    """Initializing new study and storing state and action attributes for the same.
    """
    method_name = "launch"
    logging.debug(">>>>  In %s of %s", method_name, _file_name)
    # logging.info('request.decision_config_params=%s', request.decision_config_params)

    # sorting dict key wise to maintain consistency at for all calls
    action_keys = list(request.decision_config_params.action_attrs.keys())
    action_keys.sort()
    for k in action_keys:
      self.actions[k] = request.decision_config_params.action_attrs[k]

    # sorting dict key wise to maintain consistency at for all calls
    state_keys = list(request.decision_config_params.state_attrs.keys())
    state_keys.sort()
    for k in state_keys:
      self.state[k] = request.decision_config_params.state_attrs[k]

    # sorting dict key wise to maintain consistency at for all calls
    outcome_keys = list(request.decision_config_params.outcome_attrs.keys())
    outcome_keys.sort()
    for k in outcome_keys:
      self.outcomes[k] = request.decision_config_params.outcome_attrs[k]

    print(f"<<<<<<<<<  Out {method_name} of {_file_name}.")
    logging.debug("<<<<  Out %s of %s", method_name, _file_name)
    return service_pb2.LaunchResponse()

  def decision_point(
      self, request: service_pb2.DecisionPointRequest
  ) -> service_pb2.DecisionPointResponse:
    return service_pb2.DecisionPointResponse()

  def finalize_episode(
      self, request: service_pb2.FinalizeEpisodeRequest
  ) -> service_pb2.FinalizeEpisodeResponse:
    return service_pb2.FinalizeEpisodeResponse()

  def tell(
      self, request: service_pb2.TellRequest
  ) -> service_pb2.TellResponse:
    return service_pb2.TellResponse()

  def listen(
      self, request: service_pb2.ListenRequest
  ) -> service_pb2.ListenResponse:
    return service_pb2.ListenResponse()

  def current_status(
      self, request: service_pb2.CurrentStatusRequest
  ) -> service_pb2.CurrentStatusResponse:
    return service_pb2.CurrentStatusResponse()

  def propose_action(
      self, request: service_pb2.ProposeActionRequest
  ) -> service_pb2.ProposeActionResponse:
    return service_pb2.ProposeActionResponse()

  def GetOutcome(
      self, request: service_pb2.GetOutcomeRequest
  ) -> service_pb2.GetOutcomeResponse:
    return service_pb2.GetOutcomeResponse()

  def fetch_optimal_action(
      self, request: service_pb2.FetchOptimalActionRequest
  ) -> service_pb2.FetchOptimalActionResponse:
    return service_pb2.FetchOptimalActionResponse()
