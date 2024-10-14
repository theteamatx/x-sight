import os
import sys
import unittest

from absl import app
from absl import flags
from tests.colorful_tests import ColorfulTestRunner

# Define command-line flags using absl
FLAGS = flags.FLAGS
_TYPE = flags.DEFINE_string(
    "type", None,
    "Specify the type of tests to run (e.g., 'functional', 'integration', 'performance',..)."
)
_PATTERN = flags.DEFINE_string(
    "pattern", "test_*.py",
    "Specify the file pattern to match (default is 'test_*.py').")


def discover_and_run_tests(test_type=None, pattern="test_*.py"):

  lsPaths = []

  # Find all relevant subdirectories that contain unit tests
  for path, subdirs, files in os.walk('.'):
    if "pycache" not in path and ".venv" not in path  and "tests/" in path and (test_type in path if test_type else True):
      lsPaths.append(path)

  print(f'lsPaths : {lsPaths}')

  loader = unittest.TestLoader()
  suite = unittest.TestSuite()

  for path in lsPaths:
    absolute_path = os.path.abspath(path)  # Use absolute path
    print(f'abs_path => {absolute_path}')
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    discovered = loader.discover(absolute_path, pattern=pattern)
    runner = ColorfulTestRunner(verbosity=2)
    result = runner.run(discovered)  # Run the entire suite

    if not result.wasSuccessful():
      sys.exit(1)


def main(argv):
  del argv  # Unused
  discover_and_run_tests(test_type=_TYPE.value, pattern=_PATTERN.value)


if __name__ == "__main__":
  app.run(main)
