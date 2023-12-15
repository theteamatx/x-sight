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

"""Client for exhaustive search optimizer to communicate with server."""
from typing import Optional, Sequence, Tuple
from absl import flags
from sight import service

FLAGS = flags.FLAGS


class ExhaustiveSearch:
  """Exhaustive search Client for the Sight service."""

  def __init__(self, sight):
    self._sight = sight
    self._worker_id = None

  def decision_point(self, sight, request):
    response = service.call(
        lambda s, meta: s.DecisionPoint(request, 300, metadata=meta)
    )
    print("response **************: ",response)

    return response.action[0].value.double_value

  def finalize_episode(self, sight, request):
    print('##############################################################')
    response = service.call(
        lambda s, meta: s.FinalizeEpisode(request, 300, metadata=meta)
    )
    return response
