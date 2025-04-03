"""Utility functions for converting between dictionaries and Sight protos."""

import json
from typing import Any, Callable, Dict, List, Optional

import pandas as pd
from sight.proto import sight_pb2
from sight_service.proto import service_pb2


def update_proto_map(existing_proto_map: sight_pb2.DecisionParam,
                     new_proto_map: dict[str, Any]):
  """Updates the existing proto map with the new proto map.

  Args:
    existing_proto_map: The existing proto map to update.
    new_proto_map: The new proto map to update the existing proto map with.
  """
  for key, value in new_proto_map.items():
    # Use CopyFrom to assign each new value properly
    existing_proto_map.params[key].CopyFrom(get_proto_value_from_value(value))


def get_value_from_proto_value(proto_value: sight_pb2.Value) -> Any:
  """Returns the value of the proto value.

  Args:
    proto_value: The proto value to get the value from.

  Returns: The value of the proto value.

  Raises:
    ValueError: If the proto value has an unsupported subtype.
  """
  if proto_value.sub_type == sight_pb2.Value.ST_STRING:
    return proto_value.string_value
  elif proto_value.sub_type == sight_pb2.Value.ST_BYTES:
    return proto_value.bytes_value
  elif proto_value.sub_type == sight_pb2.Value.ST_INT64:
    return proto_value.int64_value
  elif proto_value.sub_type == sight_pb2.Value.ST_DOUBLE:
    return proto_value.double_value
  elif proto_value.sub_type == sight_pb2.Value.ST_BOOL:
    return proto_value.bool_value
  elif proto_value.sub_type == sight_pb2.Value.ST_NONE:
    return None
  elif proto_value.sub_type == sight_pb2.Value.ST_JSON:
    try:
      return json.loads(proto_value.json_value)
    except (ValueError, TypeError):
      return (proto_value.json_value
             )  # Fall back to the raw string if JSON parsing fails
  else:
    raise ValueError(f"Unsupported subtype: {proto_value.sub_type}")


def get_proto_value_from_value(v) -> sight_pb2.Value:
  """Returns a proto value from a value.

  Args:
    v: The value to get the proto value from.

  Returns: The proto value of the value.

  Raises:
    ValueError: If the value has an unsupported type.
  """
  val = sight_pb2.Value()
  if isinstance(v, dict) or isinstance(v, list):
    val.sub_type = sight_pb2.Value.ST_JSON
    val.json_value = json.dumps(v)
  elif isinstance(v, pd.Series):
    val.sub_type = sight_pb2.Value.ST_JSON
    val.json_value = json.dumps(v.to_dict())
  elif isinstance(v, str):
    try:
      # Try to parse as JSON if possible
      json.loads(v)
      val.sub_type = sight_pb2.Value.ST_JSON
      val.json_value = v
    except (ValueError, TypeError):
      val.sub_type = sight_pb2.Value.ST_STRING
      val.string_value = v
  elif isinstance(v, int):
    val.sub_type = sight_pb2.Value.ST_INT64
    val.int64_value = v
  elif isinstance(v, float):
    val.sub_type = sight_pb2.Value.ST_DOUBLE
    val.double_value = v
  elif isinstance(v, bool):
    val.sub_type = sight_pb2.Value.ST_BOOL
    val.bool_value = v
  elif isinstance(v, bytes):
    val.sub_type = sight_pb2.Value.ST_BYTES
    val.bytes_value = v
  elif v is None:
    val.sub_type = sight_pb2.Value.ST_NONE
    val.none_value = True
  else:
    raise ValueError(f"Unsupported type: {type(v)}")
  return val


def convert_dict_to_proto(dict: Dict[str, Any]) -> sight_pb2.DecisionParam:
  """Converts a dictionary to a proto.

  Args:
    dict: The dictionary to convert to a proto.

  Returns:
    The proto representation of the dictionary.

  """
  proto_map = sight_pb2.DecisionParam()
  for k, v in dict.items():
    proto_map.params[k].CopyFrom(get_proto_value_from_value(v))
  return proto_map


def convert_proto_to_dict(proto: sight_pb2.DecisionParam) -> Dict[str, Any]:
  """Converts a proto to a dictionary.

  Args:
    proto: The proto to convert to a dictionary.

  Returns:
    The dictionary representation of the proto.

  """
  result = {}
  for k, v in proto.params.items():
    result[k] = get_value_from_proto_value(v)
  return result
