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

"""Access to traces of Sight-logged executions."""

from typing import Any, List, Optional, Sequence

# from google3.analysis.dremel.core.capacitor.public.python import pywrap_record_reader
from sight.proto import sight_pb2


class Trace:
  """Encapsulates execution traces.

  Attributes:
    trace_file: RecordReader that provides access to the file that contains the
      execution trace.
    trace_iter: Iterator that provides a read-once view into the trace.
    cur_obj: The current trace object. trace_iter refers to the trace segment
      that immediately follows.
  """

  def __init__(
      self,
      trace_file_path: Optional[str] = None,
      trace: Optional[Sequence[sight_pb2.Object]] = None,
  ):
    """Creates a Trace of an execution.

    Args:
      trace_file_path: Path of the file that contains the execution trace.
      trace: A concrete execution trace, used if trace_file_path is not
        provided.

    Returns:
      The object that encapsulates the trace.
    """
    if trace_file_path:
      self._trace_file = pywrap_record_reader.RecordReader.CreateFromPath(
          trace_file_path, ['*'], 60.0
      )
      log = sorted(
          list(self._trace_file.IterRecords()), key=lambda x: x.location
      )
      self._trace_iter = log.__iter__()
    else:
      self._trace_iter = iter(trace)
    self._cur_obj: Optional[sight_pb2.Object] = self._trace_iter.__next__()

  def get_cur(self) -> Optional[sight_pb2.Object]:
    """Returns the current object in the pass through the trace."""
    return self._cur_obj

  def advance_to_within_block(
      self, obj_type: Sequence[Any]
  ) -> Optional[sight_pb2.Object]:
    """Advances to the next object of a given type in the current block.

    This method focuses on singleton or start-of-block objects.
    If the initial value of self._cur_obj is an end-of-block object or if the
    end of the block is hit during the search, self._cur_obj is set
    to refer to this end-of-block object and this method returns None.

    Args:
      obj_type: Identifies the type of the objects of interest. This sequence
        contains SubType enum values, with the first belonging to
        sight.pb.Object.SubType and any subsequent values corresponding to more
        specific sub-types.

    Returns:
      The next log object of this type, if any.
    """
    if (
        self._cur_obj is None
        or self._cur_obj.sub_type == sight_pb2.Object.ST_BLOCK_END
    ):
      return None

    start_obj = self._cur_obj
    container_location = None
    if len(start_obj.ancestor_start_location) >= 2:
      container_location = start_obj.ancestor_start_location[-2]

    while True:
      if not self._cur_obj:
        return None

      if self._cur_obj.sub_type == sight_pb2.Object.ST_BLOCK_END and (
          not container_location
          or self._cur_obj.block_end.location_of_block_start
          == container_location
      ):
        target_obj = None
        # Advance the current object to the next log location.
        self._cur_obj = self._trace_iter.__next__()

      # TODO(bronevet): Replace this with a reflection-based mechanism that
      #   be automatically responsive to changes in sight.proto
      if len(obj_type) >= 1 and self._cur_obj.sub_type == obj_type[0]:
        if len(obj_type) == 1:
          target_obj = self._cur_obj
          break
        if len(obj_type) >= 2:
          if (
              obj_type[0] == sight_pb2.Object.ST_BLOCK_START
              and self._cur_obj.block_start.sub_type == obj_type[1]
          ):
            if len(obj_type) == 2:
              target_obj = self._cur_obj
              break
            if (
                len(obj_type) >= 3
                and obj_type[1] == sight_pb2.BlockStart.ST_LIST
                and self._cur_obj.block_start.list.sub_type == obj_type[2]
            ):
              target_obj = self._cur_obj
              break
            if (
                len(obj_type) >= 3
                and obj_type[1] == sight_pb2.BlockStart.ST_CONFIGURATION
                and self._cur_obj.block_start.configuration.sub_type
                == obj_type[2]
            ):
              target_obj = self._cur_obj
              break
          elif (
              obj_type[0] == sight_pb2.Object.ST_TEXT
              and self._cur_obj.text.sub_type == obj_type[1]
          ):
            target_obj = self._cur_obj
            break
          elif (
              obj_type[0] == sight_pb2.Object.ST_VALUE
              and self._cur_obj.value.sub_type == obj_type[1]
          ):
            target_obj = self._cur_obj
            break
          elif (
              obj_type[0] == sight_pb2.Object.ST_TENSOR
              and self._cur_obj.tensor.sub_type == obj_type[1]
          ):
            target_obj = self._cur_obj
            break

      self._cur_obj = self._trace_iter.__next__()

    return target_obj

  def collect_current_block(self) -> List[sight_pb2.Object]:
    """Collects the log segment that includes the current object and its block.

    If the current object is not the start of a block, the object alone is
    returned. The current object is advanced to immediately after end of the
    block or singleton object during the process of collection.

    Returns:
      The block associated with the current object.
    """
    if not self._cur_obj:
      return []

    block_start_obj = self._cur_obj
    if block_start_obj.sub_type != sight_pb2.Object.ST_BLOCK_START:
      self._cur_obj = self._trace_iter.__next__()
      return [block_start_obj]
    block_objects = []

    # Iterate until we reach an block-end object that matches block_start_obj.
    while self._cur_obj and not (
        self._cur_obj.sub_type == sight_pb2.Object.ST_BLOCK_END
        and self._cur_obj.block_end.location_of_block_start
        == block_start_obj.location
    ):
      block_objects.append(self._cur_obj)
      self._cur_obj = self._trace_iter.__next__()
    block_objects.append(self._cur_obj)

    # If we've not yet reached the end of the trace, advance it to the next
    # log object.
    if self._cur_obj:
      self._cur_obj = self._trace_iter.__next__()

    return block_objects
