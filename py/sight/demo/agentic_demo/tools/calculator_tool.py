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
"""Calculator tool that propose action to sight backend."""

import asyncio
from typing import Any, Dict

from absl import flags
from helpers.logs.logs_handler import logger as logging
from sight.sight import Sight
from sight.widgets.decision import proposal
from worker.helper import get_description_from_textproto

FLAGS = flags.FLAGS


def generate_description():
  arg_info_str = (
      "\n  The action input must contains a dictionary with key action_dict and"
      " value as dict with keys-values as follows : \n"
  )
  api_description, arguments_description = get_description_from_textproto(
      get_question_label()
  )
  description = api_description + arg_info_str + arguments_description
  logging.info("description : %s", description)
  return description


def get_question_label():
  return "Calculator"


def calculator_api(action_dict: Dict[str, Any], sight: Sight) -> str:
  """Propose actions to the server using Sight backend."""

  result = asyncio.run(
      proposal.propose_actions(sight, get_question_label(), action_dict))
  return result
