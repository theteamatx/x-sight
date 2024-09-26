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
from helpers.logs.logs_handler import logger as logging
from typing import Any, Dict, List, Tuple, Sequence
from sight_service.optimizer_instance import OptimizerInstance
from sight_service.proto import service_pb2
from sight.proto import sight_pb2

_file_name = "single_action_optimizer.py"


class SingleActionOptimizer(OptimizerInstance):
    """An SingleActionOptimizer class that is generic for all optimizers.

  An optimizer containing base methods which specialized optimizers will
  override while communicating with client.
  """

    def __init__(self):
        super().__init__()
        self.unique_id = 1
        self.pending_samples = {}
        self.active_samples = {}
        self.completed_samples = {}
