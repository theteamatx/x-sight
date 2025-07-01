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

"""Base implementation of optimizer clients."""


from abc import ABC, abstractmethod

class BaseOptimizerClient(ABC):
    """Abstract base for all optimizer clients (Bayes, NeverGrad, Queue, etc)."""
    max_reward = 0
    best_action = {}
    outcome_of_best_action = {}

    @abstractmethod
    def get_sample(self) -> dict:
        """Returns a new sample (action) to try."""
        pass

    @abstractmethod
    def document_sample(self, action: dict, reward: float, outcome: dict) -> None:
        """Stores the result of a sampled action."""
        pass
