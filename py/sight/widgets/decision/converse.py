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
import grpc
from helpers.logs.logs_handler import logger as logging
from sight import service_utils as service
from sight.proto import sight_pb2
from sight.service_utils import generate_metadata
from sight_service.proto import service_pb2
from sight_service.proto import service_pb2_grpc

_LOG_ID = flags.DEFINE_string(
    'log_id', None, 'ID of the Sight log that tracks this execution.')
_DEPLOYMENT_MODE = flags.DEFINE_enum(
    'deployment_mode',
    None,
    ['distributed', 'dsub_local', 'docker_local', 'local', 'worker_mode'],
    ('The procedure to use when training a model to drive applications that '
     'use the Decision API.'),
)


def main(argv: Sequence[str]) -> None:
  if len(argv) > 1:
    raise app.UsageError("Too many command-line arguments.")

  while True:
    message = input('# ')
    # print ('message=', message)
    req = service_pb2.TellRequest()
    req.client_id = _LOG_ID.value
    req.message_str = message
    response = service.call(lambda s, meta: s.Tell(req, 300, metadata=meta))
    print('$ ' + response.response_str)

    while True:
      req = service_pb2.ListenRequest()
      req.client_id = _LOG_ID.value
      response = service.call(lambda s, meta: s.Listen(req, 300, metadata=meta))
      if response.response_ready:
        print(response.response_str)
        break
      time.sleep(5)


if __name__ == "__main__":
  app.run(main)
