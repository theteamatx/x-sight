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
"""An instance of a Sight optimizer dedicated to a single experiment."""

from concurrent import futures
import dataclasses
from typing import Any, Dict, List, Sequence, Tuple

from helpers.logs.logs_handler import logger as logging
from sight.proto import sight_pb2
from sight_service.message_queue import IMessageQueue
from sight_service.message_queue import IncrementalUUID
from sight_service.message_queue import MessageQueue
from sight_service.optimizer_instance import OptimizerInstance
from sight_service.proto import service_pb2

_file_name = "single_action_optimizer.py"


@dataclasses.dataclass()
class MessageDetails:
  """Message details for a single message.

  Attributes:
    reward: The reward for the action.
    outcome: The outcome of the action.
    action: The action taken.
    attributes: The attributes of the action.
  """

  action: Dict[str, str]
  attributes: Dict[str, str]
  reward: float
  outcome: Dict[str, str]

  @classmethod
  def create(cls, action, attributes, reward=None, outcome=None):
    return cls(action, attributes, reward, outcome)

  def update(self, reward=None, outcome=None, action=None, attributes=None):
    if reward is not None:
        self.reward = reward
    if outcome is not None:
        self.outcome = outcome
    if action is not None:
        self.action = action
    if attributes is not None:
        self.attributes = attributes
    return self

  def __str__(self):
      return (f"MessageDetails(\n"
              f"action: {self.action},\n"
              f"attributes: {self.attributes},\n"
              f"reward: {self.reward},\n"
              f"outcome: {self.outcome}\n)")

class SingleActionOptimizer(OptimizerInstance):
  """An SingleActionOptimizer class that is generic for all optimizers.

  An optimizer containing base methods which specialized optimizers will
  override while communicating with client.
  """

  def __init__(self):
    super().__init__()
    self.queue: IMessageQueue = MessageQueue[MessageDetails](id_generator=IncrementalUUID())
