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

import dataclasses
import datetime
from typing import Dict

from helpers.logs.logs_handler import logger as logging
from sight_service.message_queue.message_logger.interface import (
    ILogStorageCollectStrategy
)
from sight_service.message_queue.message_logger.log_storage_collect import (
    CachedBasedLogStorageCollectStrategy
)
from sight_service.message_queue.message_logger.log_storage_collect import (
    NoneLogStorageCollectStrategy
)
from sight_service.message_queue.mq_interface import IMessageQueue
from sight_service.message_queue.mq_interface import IUUIDStrategy
from sight_service.message_queue.queue_factory import queue_factory
from sight_service.optimizer_instance import OptimizerInstance

datetime = datetime.datetime

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
  outcome: Dict[
      str,
      str]  # outcome is replaced by outcome_ref_key , will delete it in future
  outcome_ref_key: str = None

  @classmethod
  def create(cls, action, attributes, reward=None, outcome=None):
    return cls(action, attributes, reward, outcome)

  def update(self,
             reward=None,
             outcome=None,
             action=None,
             attributes=None,
             outcome_ref_key=None):
    if reward is not None:
      self.reward = reward
    if outcome is not None:
      self.outcome = outcome
    if action is not None:
      self.action = action
    if attributes is not None:
      self.attributes = attributes
    if outcome_ref_key is not None:
      self.outcome_ref_key = outcome_ref_key
    return self

  def __str__(self):
    return f"[X]"
    # (f"MessageDetails(\n"
    # f"action: {self.action},\n"
    # f"attributes: {self.attributes},\n"
    # f"reward: {self.reward},\n"
    # f"outcome: {self.outcome}\n)")


class SingleActionOptimizer(OptimizerInstance):
  """An SingleActionOptimizer class that is generic for all optimizers.

  An optimizer containing base methods which specialized optimizers will
  override while communicating with client.
  """

  def __init__(self, batch_size: int = 5):
    super().__init__()
    # logger_storage_strategy = NoneLogStorageCollectStrategy()
    # can use the following logger for analyis , how messages flow
    # logger_storage_strategy: ILogStorageCollectStrategy = (
    #     CachedBasedLogStorageCollectStrategy(
    #         cache_type="gcs",
    #         config={
    #             "gcs_base_dir": "sight_mq_logs_for_analysis",
    #             "gcs_bucket": "cameltrain-sight",
    #             "dir_prefix": f'log_chunks_{datetime.utcnow().strftime("%Y-%m-%d_%H-%M-%S")}/',
    #         },
    #     ))
    self.queue: IMessageQueue = queue_factory(
        queue_type="shared_lock_list",
        batch_size=batch_size,
        logger_storage_strategy=None,
    )
