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

from sight.widgets.decision.base_optimizer_client import BaseOptimizerClient
from sight.worker.worker_helper import get_action_from_textproto
from bayes_opt import BayesianOptimization
from bayes_opt import UtilityFunction


# Creating Bayesian Optimizer Object locally
def _create_bayes_opt_obj(actions):
  bayes_obj = BayesianOptimization(
      f=None,
      pbounds={
          key: (v_obj.min_value, v_obj.max_value) for key, v_obj in actions.items()
      },
      verbose=2,
      allow_duplicate_points=True,
      # random_state=1,
  )
  return bayes_obj


class BayesOptOptimizerClient(BaseOptimizerClient):

  def __init__(self, q_label_to_propose, num_questions, batch_size, sight):
    super().__init__()
    self.question_label = q_label_to_propose
    self.num_questions = num_questions
    self.batch_size = batch_size
    self.sight = sight

    actions = get_action_from_textproto(self.question_label, self.sight)
    self.bo_obj = _create_bayes_opt_obj(actions)

  def get_sample(self) -> dict:
    u_function = UtilityFunction(kind='ucb', kappa=1.96, xi=0.01)
    actions = self.bo_obj.suggest(u_function)
    return actions

  def document_sample(self, action: dict, reward: float, outcome: dict):
    self.bo_obj.register(params=action, target=reward)
    if (reward > self.max_reward):
        self.max_reward = reward
        self.best_action = action
        self.outcome_of_best_action = outcome

  def get_best_action_with_reward(self):
    return self.best_action, self.max_reward


