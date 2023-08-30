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

"""Documentation of numpy events and data in the Sight log."""

import dataclasses
import inspect
from typing import Any, List, Optional, Union

from absl import logging
import numpy as np

from sight.proto import sight_pb2
from sight.location import Location


@dataclasses.dataclass
class LabeledNpArray:
  """A variant on np.ndarrays where the dimensions are labeled."""
  array: np.ndarray

  # The labels of all the array dimensions.
  dim_label: List[str]

  # For each dimension of array contains the string labels of each slice
  # in that dimension.
  dim_axis_values: List[List[str]]


def log(label: str,
        obj_to_log: Union[np.ndarray, LabeledNpArray, np.int64],
        sight: Any,
        frame: Optional[Any] = None) -> Optional[Location]:
  """Documents numpy object in the Sight log if Sight is being used.

  Args:
    label: The label that identifies this object.
    obj_to_log: The numpy object to be logged.
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

  obj = sight_pb2.Object()

  if frame is None:
    # pytype: disable=attribute-error
    frame = inspect.currentframe().f_back
    # pytype: enable=attribute-error
  sight.set_object_code_loc(obj, frame) 

  # obj_to_log is a scalar
  if isinstance(obj_to_log, np.int64):
    obj.sub_type = sight_pb2.Object.SubType.ST_VALUE
    obj.value.sub_type = sight_pb2.Value.ST_INT64
    obj.value.int64_value = int(obj_to_log)
    return sight.log_object(obj, True)

  if isinstance(obj_to_log, np.float64):
    obj.sub_type = sight_pb2.Object.SubType.ST_VALUE
    obj.value.sub_type = sight_pb2.Value.ST_DOUBLE
    obj.value.double_value = int(obj_to_log)
    return sight.log_object(obj, True)

  if isinstance(obj_to_log, bool):
    obj.sub_type = sight_pb2.Object.SubType.ST_VALUE
    obj.value.sub_type = sight_pb2.Value.ST_BOOL
    obj.value.bool_value = int(obj_to_log)
    return sight.log_object(obj, True)

  # obj_to_log is an array
  if isinstance(obj_to_log, np.ndarray):
    labeled_array = LabeledNpArray(
        obj_to_log, [f'dim{i}' for i in range(len(obj_to_log.shape))],
        [[f'v{v}'
          for v in range(obj_to_log.shape[i])]
         for i in range(len(obj_to_log.shape))])
  elif isinstance(obj_to_log, LabeledNpArray):
    labeled_array = obj_to_log
  else:
    logging.error('Invalid type for array: %s', obj_to_log)
    return None

  obj.sub_type = sight_pb2.Object.SubType.ST_TENSOR
  obj.tensor.label = label
  obj.tensor.shape.extend(labeled_array.array.shape)
  if labeled_array.array.dtype == float or labeled_array.array.dtype == np.float32 or labeled_array.array.dtype == np.float64:
    obj.tensor.sub_type = sight_pb2.Tensor.ST_DOUBLE
    obj.tensor.double_values.value.extend(
        labeled_array.array.reshape(labeled_array.array.size).tolist())
  elif labeled_array.array.dtype == np.int or labeled_array.array.dtype == np.int32 or labeled_array.array.dtype == np.int64:
    obj.tensor.sub_type = sight_pb2.Tensor.ST_INT64
    obj.tensor.int64_values.value.extend(
        labeled_array.array.reshape(labeled_array.array.size).tolist())
  obj.tensor.dim_label.extend(labeled_array.dim_label)
  for dav in labeled_array.dim_axis_values:
    obj.tensor.dim_axis_values.append(sight_pb2.Tensor.StringValues(value=dav))

  return sight.log_object(obj, True)


def from_log(sub_log: List[sight_pb2.Object]) -> Optional[np.ndarray]:
  """Loads a numpy array from a log sub-sequence.

  Args:
    sub_log: The sub-sequence of log objects to load from.

  Returns:
    The loaded numpy array.
  """
  obj = sub_log[0]

  if obj.sub_type != sight_pb2.Object.ST_TENSOR:
    return None

  # No case for int64 since it is treated as a Python int for now
  if obj.tensor.sub_type == sight_pb2.Tensor.ST_DOUBLE:
    return np.array(obj.tensor.double_values.value).reshape(obj.tensor.shape)
  if obj.tensor.sub_type == sight_pb2.Tensor.ST_INT64:
    return np.array(obj.tensor.int64_values.value).reshape(obj.tensor.shape)

  return None
