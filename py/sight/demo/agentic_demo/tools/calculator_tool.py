from sight.widgets.decision import proposal
from typing import Any, Dict
import asyncio
from absl import flags
from sight.sight import Sight
from worker.helper import get_action_attr
from helpers.logs.logs_handler import logger as logging


FLAGS = flags.FLAGS

def generate_description():
  api_docstring = calculator_api.__doc__
  arg_info_str = "\n  The action input must contains a dictionary with key action_dict and value as dict with keys-values as follows : \n"
  arguments_description = get_action_attr(get_question_label())
  description = api_docstring  + arg_info_str + arguments_description
  logging.info("description : %s", description)
  return description


def get_question_label():
  return 'Calculator'


def calculator_api(action_dict: Dict[str, Any], sight: Sight) -> str:
  """
  Perform arithmetic operation (add, subtract, multiply, divide) using Sight backend.
  """

  result = asyncio.run(
      proposal.propose_actions(sight, get_question_label(), action_dict))
  return result
