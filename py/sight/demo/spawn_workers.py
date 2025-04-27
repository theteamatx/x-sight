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
"""Binary to spawn multiple workers with given file."""

from datetime import datetime
import math
import os
import subprocess
from typing import Any, Callable, Dict, Optional, Sequence, Text, Tuple

from absl import app
from absl import flags
from helpers.logs.logs_handler import logger as logging
from sight.proto import sight_pb2
from sight.sight import Sight
from sight.widgets.decision import trials

FLAGS = flags.FLAGS


def get_sight_instance():
  print('creating sight object')
  params = sight_pb2.Params(
      label='original_demo',
      bucket_name=f'{os.environ["PROJECT_ID"]}-sight',
  )
  sight_obj = Sight(params)
  return sight_obj


def main(argv: Sequence[str]) -> None:
  if len(argv) > 1:
    raise app.UsageError("Too many command-line arguments.")

  with get_sight_instance() as sight:
    trials.start_jobs(
        num_train_workers=1,
        binary_path='py/sight/demo/demo.py',
        optimizer_type='worklist_scheduler',
        docker_image='gcr.io/cameltrain/sight-portfolio-worker',
        decision_mode='train',
        server_mode='cloud_run',
        worker_mode='dsub_cloud_worker',
        sight=sight,
    )


if __name__ == "__main__":
  app.run(main)
