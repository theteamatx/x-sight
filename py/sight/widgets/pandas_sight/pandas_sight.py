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

from helpers.logs.logs_handler import logger as logging
import numpy as np
import pandas as pd

from sight.proto import sight_pb2
from sight.location import Location


def _df_start(
    label: str,
    sight: Any,
    frame: Any,
) -> None:
    start_obj = sight_pb2.Object()
    start_obj.sub_type = sight_pb2.Object.SubType.ST_BLOCK_START
    start_obj.block_start.sub_type = sight_pb2.BlockStart.ST_LIST
    start_obj.block_start.list.sub_type = sight_pb2.ListStart.ST_HETEROGENEOUS
    sight.enter_block(label, start_obj, frame)


def _df_end(
    label: str,
    sight: Any,
    frame: Any,
) -> None:
    end_obj = sight_pb2.Object(sub_type=sight_pb2.Object.SubType.ST_BLOCK_END)
    sight.exit_block(label, end_obj)


def log(
    label: str,
    df: pd.DataFrame,
    sight: Any,
    frame: Optional[Any] = None,
) -> Optional[Location]:
    """Documents pandas DataFrame object in the Sight log if Sight is being used.
  Args:
    label: The label that identifies this object.
    obj_to_log: The pandas frame to be logged.
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

    if frame is None:
        # pytype: disable=attribute-error
        frame = inspect.currentframe().f_back
        # pytype: enable=attribute-error

    _df_start(label, sight, frame)

    for i in range(df.shape[1]):
        nv_start_obj = sight_pb2.Object()
        nv_start_obj.sub_type = sight_pb2.Object.SubType.ST_BLOCK_START
        nv_start_obj.block_start.sub_type = sight_pb2.BlockStart.ST_NAMED_VALUE
        sight.enter_block(str(df.columns[i]), nv_start_obj, frame)

        obj = sight_pb2.Object()
        sight.set_object_code_loc(obj, frame)

        obj.sub_type = sight_pb2.Object.SubType.ST_TENSOR
        obj.tensor.label = str(df.columns[i])
        obj.tensor.shape.append(df.shape[0])
        if (df.dtypes[df.columns[i]] == float
                or df.dtypes[df.columns[i]] == np.float32
                or df.dtypes[df.columns[i]] == np.float64):
            obj.tensor.sub_type = sight_pb2.Tensor.ST_DOUBLE
            obj.tensor.double_values.value.extend(df[df.columns[i]].tolist())
        elif (
                # df.dtypes[df.columns[i]] == np.int
                # or
                df.dtypes[df.columns[i]] == np.int32
                or df.dtypes[df.columns[i]] == np.int64):
            obj.tensor.sub_type = sight_pb2.Tensor.ST_INT64
            obj.tensor.int64_values.value.extend(df[df.columns[i]].tolist())
        else:
            obj.tensor.sub_type = sight_pb2.Tensor.ST_STRING
            obj.tensor.string_values.value.extend(
                [str(v) for v in df[df.columns[i]].tolist()])
        obj.tensor.dim_label.append(str(df.columns[i]))

        sight.log_object(obj, True)

        nv_end_obj = sight_pb2.Object()
        nv_end_obj.sub_type = sight_pb2.Object.SubType.ST_BLOCK_END
        sight.exit_block(label, nv_end_obj)

    _df_end(label, sight, frame)


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
        return np.array(obj.tensor.double_values.value).reshape(
            obj.tensor.shape)
    if obj.tensor.sub_type == sight_pb2.Tensor.ST_INT64:
        return np.array(obj.tensor.int64_values.value).reshape(
            obj.tensor.shape)

    return None
