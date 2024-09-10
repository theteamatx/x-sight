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

"""Logging for data structures and base types."""

import inspect
from typing import Any, List, Optional, Sequence

import math
import numpy as np
import pandas as pd
from sight.location import Location
# import tensorflow as tf

from sight.proto import sight_pb2
from sight.widgets.decision import decision
from sight.widgets.numpy_sight import numpy_sight
from sight.widgets.pandas_sight import pandas_sight
# from py.widgets.simulation import simulation_state
# from py.widgets.tensorflow_sight import tensorflow_sight

import warnings
# Suppress FutureWarning messages
warnings.simplefilter(action='ignore', category=FutureWarning)

import warnings
# Suppress FutureWarning messages
warnings.simplefilter(action='ignore', category=FutureWarning)

class File:
  def __init__(self, path: str, mime_type: str, binary: bool=True):
    self.path = path
    self.mime_type = mime_type

def sanitize_dict(d) -> dict:
  """ preprocess data which can't be handle by the AVRO file format.
  Args:
    Dictionary to be stored in AVRO
  Returns:
    Sanitized dictionary which can be stored with AVRO
  """
  sanitized = {}
  for k, v in d.items():
    if isinstance(v, float):  # Replace NaN with None
      if v == float('inf'):
        sanitized[k] = 1e308  # Replace positive infinity
      elif v == float('-inf'):
        sanitized[k] = -1e308  # Replace negative infinity
      elif math.isnan(v):
        sanitized[k] = None
      else:
        sanitized[k] = v
    else:
      sanitized[k] = v
  return sanitized

def log_var(
    name: str, obj_to_log: Any, sight: Any, frame: Optional[Any] = None
) -> Optional[Location]:
  """Documents a named Python object, if Sight is being used.

  Args:
    name: Object's identifying name.
    obj_to_log: The object to be logged.
    sight:  The Sight object via which logging is to be done or None if Sight is
      not being used.
    frame: The call stack frame that contains the calling context information.

  Returns:
    The location of this object within the log.
  """
  if sight is None:
    return None

  if not sight.is_logging_enabled():
    return None

  sight_obj = sight_pb2.Object()

  if frame is None:
    # pytype: disable=attribute-error
    frame = inspect.currentframe().f_back
    # pytype: enable=attribute-error
  sight.set_object_code_loc(sight_obj, frame)

  sight_obj.sub_type = sight_pb2.Object.SubType.ST_BLOCK_START
  sight_obj.block_start.sub_type = sight_pb2.BlockStart.ST_NAMED_VALUE

  sight.enter_block(name, sight_obj, frame)

  # TODO(bronevet): this is an unmaintainable mechanism for informing modules
  #   about changes to state. Will need to have a generic observer mechanism.
  decision.state_updated(name, obj_to_log, sight)
  # simulation_state.state_updated(name, obj_to_log, sight)

  log(obj_to_log, sight, frame)

  end_obj = sight_pb2.Object()
  end_obj.sub_type = sight_pb2.Object.SubType.ST_BLOCK_END
  sight.exit_block(name, end_obj)


def log(
    obj_to_log: Any, sight: Any, frame: Optional[Any] = None
) -> Optional[Location]:
  """Documents any Python object, if Sight is being used.

  Args:
    obj_to_log: The object to be logged.
    sight:  The Sight object via which logging is to be done or None if Sight is
      not being used.
    frame: The call stack frame that contains the calling context information.

  Returns:
    The location of this object within the log.
  """

  if sight is None:
    return None

  if not sight.is_logging_enabled():
    return None

  sight_obj = sight_pb2.Object()

  if frame is None:
    # pytype: disable=attribute-error
    frame = inspect.currentframe().f_back
    # pytype: enable=attribute-error
  sight.set_object_code_loc(sight_obj, frame)

  if isinstance(obj_to_log, str):
    sight_obj.sub_type = sight_pb2.Object.SubType.ST_VALUE
    sight_obj.value.sub_type = sight_pb2.Value.ST_STRING
    sight_obj.value.string_value = obj_to_log
    sight.log_object(sight_obj, True)
  elif isinstance(obj_to_log, bytes):
    sight_obj.sub_type = sight_pb2.Object.SubType.ST_VALUE
    sight_obj.value.sub_type = sight_pb2.Value.ST_BYTES
    sight_obj.value.bytes_value = obj_to_log
    sight.log_object(sight_obj, True)
  elif isinstance(obj_to_log, int):
    sight_obj.sub_type = sight_pb2.Object.SubType.ST_VALUE
    sight_obj.value.sub_type = sight_pb2.Value.ST_INT64
    sight_obj.value.int64_value = obj_to_log
    sight.log_object(sight_obj, True)
  elif isinstance(obj_to_log, float):
    sight_obj.sub_type = sight_pb2.Object.SubType.ST_VALUE
    sight_obj.value.sub_type = sight_pb2.Value.ST_DOUBLE
    sight_obj.value.double_value = obj_to_log
    sight.log_object(sight_obj, True)
  elif isinstance(obj_to_log, bool):
    sight_obj.sub_type = sight_pb2.Object.SubType.ST_VALUE
    sight_obj.value.sub_type = sight_pb2.Value.ST_BOOL
    sight_obj.value.bool_value = obj_to_log
    sight.log_object(sight_obj, True)
  elif obj_to_log is None:
    sight_obj.sub_type = sight_pb2.Object.SubType.ST_VALUE
    sight_obj.value.sub_type = sight_pb2.Value.ST_NONE
    sight_obj.value.none_value = True
    sight.log_object(sight_obj, True)
  elif isinstance(obj_to_log, File):
    sight_obj.sub_type = sight_pb2.Object.SubType.ST_VALUE
    with open(obj_to_log.path, mode='rb') as f:
      sight_obj.value.sub_type = sight_pb2.Value.ST_BYTES
      sight_obj.value.string_value = f.read()
      #sight_obj.value.bytes_value = f.read()
      sight_obj.value.mime_type = obj_to_log.mime_type
      sight.log_object(sight_obj, True)
  elif (
      isinstance(obj_to_log, list)
      or isinstance(obj_to_log, tuple)
      or isinstance(obj_to_log, set)
  ):
    sight_obj.sub_type = sight_pb2.Object.SubType.ST_BLOCK_START
    sight_obj.block_start.sub_type = sight_pb2.BlockStart.ST_LIST
    sight_obj.block_start.list.sub_type = sight_pb2.ListStart.ST_HETEROGENEOUS

    if isinstance(obj_to_log, list):
      label = 'list'
    elif isinstance(obj_to_log, tuple):
      label = 'tuple'
    else:
      label = 'set'
    sight.enter_block(label, sight_obj, frame)

    for element in obj_to_log:
      log(element, sight, frame)

    end_obj = sight_pb2.Object(sub_type=sight_pb2.Object.SubType.ST_BLOCK_END)
    sight.exit_block(label, end_obj)
  elif (
      isinstance(obj_to_log, np.int64)
      or isinstance(obj_to_log, np.float64)
      or isinstance(obj_to_log, bool)
  ):
    numpy_sight.log('np scalar', obj_to_log, sight, frame)
  elif isinstance(obj_to_log, np.ndarray) or isinstance(
      obj_to_log, numpy_sight.LabeledNpArray
  ):
    numpy_sight.log('np array', obj_to_log, sight, frame)
  # elif isinstance(obj_to_log, tf.Tensor):
  #   tensorflow_sight.log('TensorFlow Tensor', obj_to_log, sight, frame)
  elif isinstance(obj_to_log, pd.DataFrame):
    pandas_sight.log('pandas.DataFrame', obj_to_log, sight, frame)
  elif isinstance(obj_to_log, dict) or hasattr(obj_to_log, '__dict__'):
    if isinstance(obj_to_log, dict):
      if(sight.params.file_format == '.avro'):
        # need to handle inf/NaN values in dictionary for avro
        obj_dict = sanitize_dict(obj_to_log)
      else:
        obj_dict = obj_to_log
    else:
      obj_dict = obj_to_log.__dict__
    sight_obj.sub_type = sight_pb2.Object.SubType.ST_BLOCK_START
    sight_obj.block_start.sub_type = sight_pb2.BlockStart.ST_LIST
    sight_obj.block_start.list.sub_type = sight_pb2.ListStart.ST_MAP
    sight.enter_block('map', sight_obj, frame)

    for key, value in obj_dict.items():
      item_start = sight_pb2.Object()
      item_start.sub_type = sight_pb2.Object.SubType.ST_BLOCK_START
      item_start.block_start.sub_type = sight_pb2.BlockStart.ST_LIST
      item_start.block_start.list.sub_type = sight_pb2.ListStart.ST_MAP_ENTRY
      sight.enter_block('map_entry', item_start, frame)

      # logging.info('logging key=%s/%s value=%s/%s', key, type(key), value,
      #              type(value))
      log(key, sight, frame)
      log(value, sight, frame)

      item_end_obj = sight_pb2.Object(
          sub_type=sight_pb2.Object.SubType.ST_BLOCK_END,
          block_start=sight_pb2.BlockStart(
              sub_type=sight_pb2.BlockStart.ST_LIST
          ),
      )
      sight.exit_block('map_entry', item_end_obj)

    end_obj = sight_pb2.Object(
        sub_type=sight_pb2.Object.SubType.ST_BLOCK_END,
        block_start=sight_pb2.BlockStart(sub_type=sight_pb2.BlockStart.ST_LIST),
    )
    sight.exit_block('map', end_obj)
  else:
    # print('OTHER')
    sight_obj.sub_type = sight_pb2.Object.SubType.ST_VALUE
    sight_obj.value.sub_type = sight_pb2.Value.ST_STRING
    sight_obj.value.string_value = str(obj_to_log)
    sight.log_object(sight_obj, True)


def get_full_sublog_of_first_element(
    log_segment: Sequence[sight_pb2.Object],
) -> List[sight_pb2.Object]:
  """Returns the section of an ordered log that corresponds to the first element.

  If the first element is a BlockStart, this function searches the log for the
  matching BlockEnd and returns the sub-sequence that includes them and the
  objects between them. Otherwise, this function returns just a list containing
  the first element.

  Args:
    log_segment: ordered segment of a Sight log.

  Returns:
    Sub-sequence of log that corresponds to the first element of log and any
    of its contained objects.
  """
  # If log_segment[0] demarcates the start of a block.
  if log_segment[0].sub_type == sight_pb2.Object.ST_BLOCK_START:
    # Find the end of this block.
    depth = 1
    for j in range(1, len(log_segment)):
      if log_segment[j].sub_type == sight_pb2.Object.ST_BLOCK_START:
        depth += 1
      elif log_segment[j].sub_type == sight_pb2.Object.ST_BLOCK_END:
        depth -= 1
        if depth == 0:
          return list(log_segment[0 : j + 1])

  return list(log_segment[0:1])


def from_log(log_segment: List[sight_pb2.Object]) -> Any:
  """Creates an object from an unordered sub-segment of a Sight log."""
  log_segment.sort(key=lambda x: x.location)
  return from_ordered_log(log_segment)


def from_ordered_log(log_segment: List[sight_pb2.Object]) -> Any:
  """Creates an object from an ordered sub-segment of a Sight log."""
  if not log_segment:
    return None

  start = log_segment[0]
  if start.sub_type == sight_pb2.Object.SubType.ST_VALUE:
    if start.value.sub_type == sight_pb2.Value.ST_STRING:
      return start.value.string_value
    if start.value.sub_type == sight_pb2.Value.ST_BYTES:
      return start.value.bytes_value
    if start.value.sub_type == sight_pb2.Value.ST_INT64:
      return start.value.int64_value
    if start.value.sub_type == sight_pb2.Value.ST_DOUBLE:
      return start.value.double_value
    if start.value.sub_type == sight_pb2.Value.ST_BOOL:
      return start.value.bool_value
    if start.value.sub_type == sight_pb2.Value.ST_NONE:
      return None
  elif start.sub_type == sight_pb2.Object.SubType.ST_BLOCK_START:
    sub_log = log_segment[1 : len(log_segment) - 1]
    if start.block_start.sub_type == sight_pb2.BlockStart.ST_LIST:
      if (
          start.block_start.list.sub_type
          == sight_pb2.ListStart.ST_HETEROGENEOUS
      ):
        list_objects = []
        i = 0
        while i < len(sub_log):
          cur = get_full_sublog_of_first_element(sub_log[i : len(sub_log)])
          list_objects.append(from_ordered_log(cur))
          i += len(cur)

        if start.block_start.label == 'list':
          return list_objects
        elif start.block_start.label == 'tuple':
          return tuple(list_objects)
        elif start.block_start.label == 'set':
          return set(list_objects)
      elif start.block_start.list.sub_type == sight_pb2.ListStart.ST_MAP_ENTRY:
        key_sub_log = get_full_sublog_of_first_element(
            sub_log[0 : len(sub_log)]
        )
        value_sub_log = get_full_sublog_of_first_element(
            sub_log[len(key_sub_log) : len(sub_log)]
        )
        return (from_ordered_log(key_sub_log), from_ordered_log(value_sub_log))
      elif start.block_start.list.sub_type == sight_pb2.ListStart.ST_MAP:
        map_object = {}
        i = 0
        while i < len(sub_log) - 1:
          key_value_sub_log = get_full_sublog_of_first_element(
              sub_log[i : len(sub_log)]
          )
          (key, value) = from_ordered_log(key_value_sub_log)
          map_object[key] = value
          i += len(key_value_sub_log)
        return map_object
    elif start.block_start.sub_type == sight_pb2.BlockStart.ST_NAMED_VALUE:
      return (start.block_start.label, from_ordered_log(sub_log))
  elif start.sub_type == sight_pb2.Object.ST_TENSOR:
    return numpy_sight.from_log([start])

  return None
