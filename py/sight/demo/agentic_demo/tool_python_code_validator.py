from langchain.tools import tool
import ast
import re


@tool
def validate_python_code(raw_code_string: str) -> str:
  """
    Use this tool to validate a string of Python code.
    It cleans the code by removing markdown fences, then checks for syntax errors.
    Input must be the raw string containing the code.
    It returns the cleaned, valid code if successful, or a descriptive error message if the syntax is invalid.
    """
  # Extract code from markdown block
  match = re.search(r"```python\n(.*?)```", raw_code_string, re.DOTALL)
  if match:
      code = match.group(1).strip()
  else:
    code = raw_code_string.strip()

  if not code:
    return "Error: No code provided in the string."

  # Validate syntax using ast.parse()
  try:
    ast.parse(code)
    return f"Validation successful. The following code is syntactically valid:\n\n{code}"
  except SyntaxError as e:
    return f"Validation Failed. The code has a syntax error: {e}"
