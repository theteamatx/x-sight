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
"""Worker script to be run via FVS problem."""
import random
from typing import Tuple

from absl import app
from sight import sight


def get_question_label():
  return 'Fvs'


def main(action: dict) -> Tuple[float, dict]:
  mitigation_list = [
      227.6,
      273.4,
      273.3,
      248.6,
      165.3,
      130.6,
      106.4,
      92.1,
      81.7,
      62.8,
  ]
  random_reward = random.uniform(2, 2)

  reward = random_reward
  outcome = {'time_series': mitigation_list}

  return reward, outcome


if __name__ == '__main__':
  app.run(lambda _: sight.run_worker(
      main,
      {
          'label': get_question_label(),
      },
  ))
