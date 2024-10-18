"""Discovering and running tests."""

import os
import sys
import unittest

from absl import app
from absl import flags
from tests.colorful_tests import ColorfulTestRunner

# Define command-line flags using absl
FLAGS = flags.FLAGS
_TEST_TYPE = flags.DEFINE_string(
    "type",
    None,
    "Specify the type of tests to run (e.g., 'functional', 'integration',"
    " 'performance',..).",
)
_FILE_PATTERN = flags.DEFINE_string(
    "pattern",
    "test_*.py",
    "Specify the file pattern to match (default is 'test_*.py').",
)


def discover_and_run_tests(test_type=None, pattern="test_*.py"):
  """Discover and run tests.

  Args:
      test_type: The type of tests to run (e.g., 'functional', 'integration',
        'performance').
      pattern: The file pattern to match for test files (default is
        'test_*.py').
  """

  ls_paths = []

  # Walk through all directories and subdirectories starting from the current
  # directory.
  for path, _, _ in os.walk("."):
    # Filter out paths that contain 'pycache' or virtual environment
    # directories. Also, ensure that the path contains a 'tests/' directory and
    # optionally match the 'test_type' if specified.
    if (
        "pycache" not in path
        and ".venv" not in path
        and "tests/" in path
        and (test_type in path if test_type else True)
    ):
      # Add the path to the list of discovered test directories.
      ls_paths.append(path)

  # Print the list of discovered test paths (for debugging purposes).
  print(f"ls_paths : {ls_paths}")

  # Iterate through each discovered path containing test cases.
  for path in ls_paths:
    # Convert the relative path to an absolute path for clarity and reliability.
    absolute_path = os.path.abspath(path)
    print(f"abs_path => {absolute_path}")

    # Create a test loader to find and load test cases based on the specified
    # pattern.
    loader = unittest.TestLoader()
    discovered = loader.discover(absolute_path, pattern=pattern)

    # Run the discovered test cases using the custom ColorfulTestRunner.
    runner = ColorfulTestRunner(verbosity=2)
    result = runner.run(discovered)

    # If any test fails, exit with a status code of 1 to indicate failure.
    if not result.wasSuccessful():
      sys.exit(1)


def main(argv):
  del argv  # Unused
  # Call the function with values obtained from the command-line flags.
  discover_and_run_tests(test_type=_TEST_TYPE.value, pattern=_FILE_PATTERN.value)


if __name__ == "__main__":
  app.run(main)
