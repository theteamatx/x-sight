from langchain_core.tools import Tool, tool
from sight.widgets.decision import proposal
from typing import Any
from typing_extensions import Annotated
import asyncio
from absl import flags
from sight.sight import Sight
from langchain_core.tools import InjectedToolArg, tool
from pydantic import BaseModel, Field  # <--- For explicit args_schema

FLAGS = flags.FLAGS


def get_question_label():
  return 'Calculator'


class CalculatorToolArgs(BaseModel):
  a: int = Field(description="The first integer operand.")
  b: int = Field(description="The second integer operand.")
  ops: str = Field(
      description=
      "The operation to perform. Must be one of: 'add', 'subtract', 'multiply', 'divide'."
  )


def calculator_api(a: int, b: int, ops: str, sight: Sight) -> str:
  """
  Perform a basic arithmetic operation (addition, subtraction, etc.) on two integers using the Sight backend system.

  This function proposes a calculation action (with inputs `a`, `b`, and the operation `ops`) to the server via a Sight worker.
  It waits for the worker to process the action and return the computed result.

  Args:
      a (int): The first integer operand.
      b (int): The second integer operand.
      ops (str): The operation to perform. Supported operations include:
          - "add" for addition
          - "subtract" for subtraction
          - "multiply" for multiplication
          - "divide" for division

  Returns:
      str: The result of the calculation as a string.

  Example:
      >>> calculator_api_with_sight(5, 3, "add")
      "8"
  """

  actions = {"v1": a, "v2": b, "ops": ops}
  result = asyncio.run(
      proposal.propose_actions(sight, get_question_label(), actions))
  return result
