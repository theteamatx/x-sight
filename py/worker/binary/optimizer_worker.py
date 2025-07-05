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
from typing import Tuple, Dict
import warnings

import os
from absl import app
from absl import flags
from sight import sight
from sight.sight import Sight
from sight.widgets.decision import proposal
from sight.widgets.decision.optimizers.bayes_opt_client import BayesOptOptimizerClient


def warn(*args, **kwargs):
  pass


warnings.warn = warn

FLAGS = flags.FLAGS


def get_question_label_to_propose_actions():
  return "Fvs"


def get_question_label():
  return "Generic"


# def main(sight: Sight, action: Dict[str, int], is_last_action: bool) -> Tuple[float, Dict[str, int]]:

#   # using actions we received from optimizer to propose actions to
#   # worklist_scheduler
#   outcome = asyncio.run(
#       proposal.propose_actions(sight,
#                                get_question_label_to_propose_actions(),
#                                action_dict=action,
#                                is_last_action=is_last_action))

#   vals = list(action.values())
#   # some mechanchism to calculate reward from the response of WS worker
#   reward = sum(xi**2 for xi in vals)

#   return reward, outcome


async def propose_parallely(sight: Sight, opt_obj):
  num_questions = int(os.getenv('NUM_QUESTIONS', '1'))
  batch_size = 5
  is_last_action = False
  # tasks = []
  actions = []
  rewards = []
  outcomes = []

  for i in range(0, num_questions, batch_size):

    batch_actions = []
    batch_outcome = []
    tasks = []
    # proposing in batch actions
    for itr in range(i, min(i+batch_size, num_questions)):
      if itr == num_questions - 1:
        is_last_action = True

      action = opt_obj.get_sample()
      batch_actions.append(action)
      tasks.append(
          sight.create_task(
              proposal.propose_actions(sight,
                                      get_question_label_to_propose_actions(),
                                      action_dict=action)))

    print('batch_actions : ', batch_actions, len(batch_actions))
    # wait for their output, update optimizer
    batch_outcome = await asyncio.gather(*tasks)
    print('batch_outcome : ', batch_outcome, len(batch_outcome))
    vals = list(batch_actions)
    # some mechanchism to calculate reward from the response of WS worker
    reward = 100 #static

    for b in range(len(batch_outcome)):
      outcomes.append(batch_outcome[b])
      rewards.append(reward)
      actions.append(batch_actions[b])

      #some way to document all the actions with its outcome
      opt_obj.document_sample(batch_actions[b], reward, batch_outcome[b])

  # return all actions with its rewards, outcomes
  return actions, rewards, outcomes


def main(sight: Sight) -> Tuple[float, Dict[str, int]]:

  opt_obj = BayesOptOptimizerClient(sight)
  actions, rewards, outcomes = asyncio.run(propose_parallely(sight, opt_obj))

  return actions, rewards, outcomes


if __name__ == "__main__":
  app.run(lambda _: sight.run_worker(
      main,
      {
          "label": get_question_label(),
      },
      True
  ))
