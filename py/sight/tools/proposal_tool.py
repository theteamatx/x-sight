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
"""Proposal tool that propose relavant actions to sight backend."""

import asyncio
from typing import Any, Dict

from sight.sight import Sight
from sight.widgets.decision import proposal


def proposal_api(action_dict: Dict[str, Any], sight: Sight, question_label: str) -> str:
  """Propose actions to the server using Sight backend."""

  result = asyncio.run(
      proposal.propose_actions(sight, question_label, action_dict))
  return result
