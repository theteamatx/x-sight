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
"""Documentation of exception events in the sight log."""

from helpers.logs.logs_handler import logger as logging
from sight.proto import sight_pb2


def exception(exc_type, value, traceback, sight, frame):
  """Documents an exception events in the Sight log if Sight is being used.

  Args:
    exc_type: The exc_type of the exception that was thrown
    value: The value associated with the exception.
    traceback: The stack trace of the exception.
    sight:  The Sight object via which logging is to be done or None if Sight is
      not being used.
    frame: The call stack frame that contains the calling context information.
  """
  logging.exception(
      'Exception: exc_type=%s, value=%s, traceback=%s',
      str(exc_type),
      str(value),
      str(traceback),
  )
  if sight is not None:
    sight.enter_block('Exception', sight_pb2.Object(), frame=frame)
    sight.text_block('exc_type', str(exc_type), frame=frame)
    sight.text_block('value', str(value), frame=frame)
    sight.text_block('traceback', str(traceback), frame=frame)
    sight.exit_block('Exception', sight_pb2.Object(), frame=frame)
