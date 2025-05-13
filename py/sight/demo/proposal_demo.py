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
"""Demo of using the Sight Propose action API to add actions to server and wait for it's outcome."""
import warnings


def warn(*args, **kwargs):
  pass


warnings.warn = warn

import asyncio
from typing import Sequence, Any

from absl import app
from absl import flags
from sight.sight import Sight
from sight.widgets.decision import decision
from sight.widgets.decision import proposal
from helpers.logs.logs_handler import logger as logging

FLAGS = flags.FLAGS


def get_question_label_to_propose_actions():
  return 'calculator'


def main(argv: Sequence[str]) -> None:
  if len(argv) > 1:
    raise app.UsageError("Too many command-line arguments.")

  # config contains the data from all the config files
  config = decision.DecisionConfig(config_dir_path=FLAGS.config_path)

  # create sight object with configuration to spawn workers beforehand
  with Sight.create("calculator_demo", config) as sight:

    #Ideally this actions will be proposed from some other module
    actions = {"v1": 3, "v2": 5, "ops": 'multiply'}

    asyncio.run(proposal.propose_actions(sight, get_question_label_to_propose_actions(), actions))


if __name__ == "__main__":
  app.run(main)
