from langchain_core.tools import Tool, tool
from sight.widgets.decision import proposal
from typing import Any, Dict
from typing_extensions import Annotated
import asyncio
from absl import flags
from sight.sight import Sight
from langchain_core.tools import InjectedToolArg, tool
from pydantic import BaseModel, Field  # <--- For explicit args_schema

FLAGS = flags.FLAGS

# calculator_args_schema_json = {
#     "type": "object",
#     "properties": {
#         "action_dict": {
#             "type":
#                 "object",
#             "description":
#                 "Dictionary of action parameters for the calculator.",
#             "properties": {
#                 "v1": {
#                     "type": "integer",
#                     "description": "The first integer operand."
#                 },
#                 "v2": {
#                     "type": "integer",
#                     "description": "The second integer operand."
#                 },
#                 "ops": {
#                     "type":
#                         "string",
#                     "description":
#                         "The operation to perform. Must be one of: 'add', 'subtract', 'multiply', 'divide'.",
#                     "enum": ["add", "subtract", "multiply",
#                              "divide"]  # Using 'enum' for allowed values
#                 }
#             },
#             "required": ["v1", "v2", "ops"]
#         }
#     },
#     "required": ["action_dict"]
# }


def get_question_label():
  return 'Calculator'


class CalculatorToolArgs(BaseModel):
  action_dict: Dict[str,
                    Any] = Field(...,
                                 description="Dictionary of action parameters.")


def calculator_api(action_dict: Dict[str, Any], sight: Sight) -> str:

  result = asyncio.run(
      proposal.propose_actions(sight, get_question_label(), action_dict))
  return result
