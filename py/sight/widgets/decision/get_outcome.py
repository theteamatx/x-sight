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

from typing import Any, Callable, Dict, Optional, Sequence, Text, Tuple

from absl import app
from absl import flags
from helpers.logs.logs_handler import logger as logging
from sight import service_utils as service
from sight_service.proto import service_pb2
from sight_service.proto import service_pb2_grpc

FLAGS = flags.FLAGS

_LOG_ID = flags.DEFINE_string(
    "log_id", None, "ID of the Sight log that tracks this execution.")
_SERVER_MODE = flags.DEFINE_enum(
    'server_mode',
    None,
    ['vm', 'cloud_run', 'local'],
    ('The procedure to use when training a model to drive applications that '
     'use the Decision API.'),
)


def main(argv: Sequence[str]) -> None:
  if len(argv) > 1:
    raise app.UsageError("Too many command-line arguments.")

  request = service_pb2.GetOutcomeRequest()
  request.client_id = str(FLAGS.log_id)
  # request.unique_ids.append(1)
  response = service.call(
      lambda s, meta: s.GetOutcome(request, 300, metadata=meta))

  if (response.response_str):
    return response.response_str

  outcome_list = []
  for outcome in response.outcome:
    outcome_dict = {}
    outcome_dict['reward'] = outcome.reward
    outcome_dict['action'] = dict(outcome.action_attrs)
    outcome_dict['outcome'] = dict(outcome.outcome_attrs)
    outcome_list.append(outcome_dict)
  return outcome_list


if __name__ == "__main__":
  app.run(main)
