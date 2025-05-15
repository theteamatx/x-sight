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
"""Generic worker which propose to worklist_scheduler optimizer."""
import warnings


def warn(*args, **kwargs):
  pass


warnings.warn = warn

import asyncio
import inspect
import json
import os
import random
from typing import Sequence, Any, Tuple

from absl import app
from absl import flags
import numpy as np
import pandas as pd
from sight.attribute import Attribute
from sight.block import Block
from sight import data_structures
from sight.proto import sight_pb2
from sight import sight
from sight.sight import Sight
from sight.widgets.decision import decision
from sight.widgets.decision import proposal
from helpers.logs.logs_handler import logger as logging
from helpers.decorators.decision_worker import decision_worker

FLAGS = flags.FLAGS


def get_question_label_to_propose_actions():
  return 'FVS'


def get_question_label():
  return 'generic'


def main(sight: Sight, action: dict) -> Tuple[float, dict]:

  # using actions we received from optimizer to propose actions to worklist_scheduler
  outcome = asyncio.run(
      proposal.propose_actions(sight,
                               get_question_label_to_propose_actions(),
                               action_dict=action))

  vals = list(action.values())
  # some mechanchism to calculate reward from the response of WS worker
  reward = sum(xi**2 for xi in vals)

  return reward, outcome


if __name__ == "__main__":
  app.run(lambda _: sight.run_worker(main, {
      'label': get_question_label(),
  }))
