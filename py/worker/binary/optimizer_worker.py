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

import asyncio
from typing import Tuple, Dict, Any
import warnings

import os
from absl import app
from absl import flags
from sight import sight
from sight.sight import Sight
from sight.widgets.decision import proposal
from helpers.logs.logs_handler import logger as logging
from sight.widgets.decision.optimizers.bayes_opt_client import BayesOptOptimizerClient


def warn(*args, **kwargs):
  pass


warnings.warn = warn

FLAGS = flags.FLAGS

def get_question_label():
  return "Generic"


async def optimize(sight: Sight, opt_obj):

  rewards = []
  outcomes = []

  for i in range(0, opt_obj.num_questions, opt_obj.batch_size):
    batch_actions = []
    batch_outcome = []
    tasks = []
    # proposing in batch actions
    for itr in range(i, min(i+opt_obj.batch_size, opt_obj.num_questions)):
      action = opt_obj.get_sample()
      batch_actions.append(action)
      tasks.append(
          sight.create_task(
              proposal.propose_actions(sight,
                                      opt_obj.question_label,
                                      action_dict=action)))

    # wait for their output, update optimizer
    batch_outcome = await asyncio.gather(*tasks)
    for b in range(len(batch_outcome)):
      outcomes.append(batch_outcome[b])
      # TODO(user): Implement a mechanism to calculate reward from the response of the worker.
      reward = 100 #static
      rewards.append(reward)

      # document all the actions with its outcome
      opt_obj.document_sample(batch_actions[b], reward, batch_outcome[b])

  final_outcome = {"reward" : rewards, "outcome": outcomes}
  # return all actions with its rewards, outcomes
  return final_outcome


def main(sight: Sight, action: Dict) -> Tuple[float, Dict[str, Any]]:

  # Here action will be containing optimizer config to create opt obj
  opt_obj = BayesOptOptimizerClient(action)
  final_outcome = asyncio.run(optimize(sight, opt_obj))

  # keeping reward fixed (0) as action contains optimizer config and not actual action attrs
  return 0, final_outcome


if __name__ == "__main__":
  app.run(lambda _: sight.run_worker(
      main,
      {
          "label": get_question_label(),
      }
  ))
