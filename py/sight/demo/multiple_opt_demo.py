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
"""Demo of spawning multiple worker which can interact with each other."""

from typing import Sequence
import warnings

from absl import app
from absl import flags
from helpers.logs.logs_handler import logger as logging
from sight.sight import Sight
from sight.widgets.decision import decision


def warn(*args, **kwargs):
  pass


warnings.warn = warn

FLAGS = flags.FLAGS


def main(argv: Sequence[str]) -> None:
  if len(argv) > 1:
    raise app.UsageError("Too many command-line arguments.")

  config = decision.DecisionConfig(config_dir_path=FLAGS.config_path)

  # Sight parameters dictionary with valid key values from sight_pb2.Params
  params = {"label": "multiple_opt_label"}

  # create sight object with configuration to spawn workers beforehand
  with Sight.create(params, config):

    logging.info("spawned the workers.................")


if __name__ == "__main__":
  app.run(main)
