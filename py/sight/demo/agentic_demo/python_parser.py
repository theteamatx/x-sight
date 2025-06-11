import ast
import re
from typing import Any
from langchain_core.output_parsers import BaseOutputParser
from langchain_core.exceptions import OutputParserException


def _clean_python_code(code_string: str) -> str:
  """Cleans a string to extract raw Python code, removing Markdown fences."""
  if not isinstance(code_string, str):
    raise OutputParserException(
        f"Input to parser was not a string, but {type(code_string)}.",
        llm_output=str(code_string)  # Pass the problematic output
    )
  match = re.search(r"^\s*```(?:python)?\s*\n(.*?)\n\s*```\s*$", code_string,
                    re.DOTALL)
  if match:
    return match.group(1).strip()
  return code_string.strip()


def _is_valid_python(code_string: str) -> bool:
  """Checks if a string contains syntactically valid Python code."""
  try:
    ast.parse(code_string)
    return True
  except SyntaxError:
    return False


class PythonCodeParser(BaseOutputParser):
  """
    Parses the LLM output to extract and validate a raw Python code string.

    - Cleans markdown fences.
    - Validates the syntax using ast.parse().
    - Raises OutputParserException on failure.
    """

  def parse(self, text: str) -> str:
    """
        Parses the LLM output string, returning validated Python code.
        """
    # Clean the code to remove backticks
    cleaned_code = _clean_python_code(text)

    # Validate the syntax of the cleaned code
    if _is_valid_python(cleaned_code):
      return cleaned_code
    else:
      # If invalid, raise an exception with the problematic output.
      # LangChain can use this for retry logic.
      raise OutputParserException(
          "The generated code is not valid Python syntax.",
          llm_output=text  # Pass the original, uncleaned text for context
      )

  # def get_format_instructions(self) -> str:
  #   return ("Your output must be a single, raw Python code block. "
  #           "Do not include Markdown fences like '```python'. "
  #           "The code must be syntactically correct and ready for execution.")
