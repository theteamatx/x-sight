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
"""Binary that checks the current status of a given Sight optimization run."""

import inspect
import os
import sys
import time
from typing import Any, Callable, Dict, Optional, Sequence, Text, Tuple

from absl import app
from absl import flags
from helpers.logs.logs_handler import logger as logging
import grpc
from sight_service.proto import service_pb2
from sight_service.proto import service_pb2_grpc
from sight import service_utils as service

from sight.proto import sight_pb2
from sight.service_utils import generate_metadata

_LOG_ID = flags.DEFINE_string(
    "log_id", None, "ID of the Sight log that tracks this execution."
)


def main(argv: Sequence[str]) -> None:
  if len(argv) > 1:
    raise app.UsageError("Too many command-line arguments.")

  req = service_pb2.CurrentStatusRequest()
  req.client_id = _LOG_ID.value
  response = service.call(
      lambda s, meta: s.CurrentStatus(req, 300, metadata=meta))

  if response.status == service_pb2.CurrentStatusResponse.Status.DEFAULT :
    print('Experiment is in Default state')
  elif response.status == service_pb2.CurrentStatusResponse.Status.IN_PROGRESS :
    print('Experiment is in-progress state')
  elif response.status == service_pb2.CurrentStatusResponse.Status.SUCCESS :
    print('Experiment is in Success state')
  elif response.status == service_pb2.CurrentStatusResponse.Status.FAILURE :
    print('Experiment is in Failure state')
  else:
    print('response.status = ', response.status)
  print('response :', response.response_str)


if __name__ == "__main__":
    app.run(main)
