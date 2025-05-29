from helpers.logs.logs_handler import logger as logging
from langchain_core.tools import StructuredTool
from typing import Any, Dict
from sight.worker.worker_helper import get_description_from_textproto


def generate_description(question_label) -> str:
  arg_info_str = (
      "\n  The action input must contains a dictionary with only one key `action_dict` and"
      " it's corresponding value as dictionary with keys-values as follows : \n"
  )
  api_description, arguments_description = get_description_from_textproto(
      question_label)
  description = api_description + arg_info_str + arguments_description
  logging.info("description : %s", description)
  return description


def create_tool_with_sight(sight, question_label, tool_fn) -> StructuredTool:

  def tool_fn_with_sight(action_dict: Dict[str, Any]):
    return tool_fn(action_dict=action_dict,
                   sight=sight,
                   question_label=question_label)

  # tool object to be used by agent
  tool_with_sight = StructuredTool.from_function(
      name="sight_tool",
      func=tool_fn_with_sight,
      verbose=True,
      description=generate_description(question_label),
  )

  return tool_with_sight
