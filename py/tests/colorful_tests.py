"""This module contains test cases"""

import unittest

import colorama
from helpers.logs.logs_handler import logger as logging

init = colorama.init
Style = colorama.Style
Fore = colorama.Fore

# Initialize colorama
init(autoreset=True)


class ColorfulTestResult(unittest.TextTestResult):
  """A TextTestResult class that adds color to the output.
  """

  def addSuccess(self, test):
    super().addSuccess(test)
    logging.info('%s test-cases passed', test)
    self.stream.write('\n' + Fore.GREEN + 'PASS' + Style.RESET_ALL + '\n')

  def addFailure(self, test, err):
    super().addFailure(test, err)
    logging.info('%s test-cases failed , err %s', test, err)
    self.stream.write('\n' + Fore.RED + 'FAIL' + Style.RESET_ALL + '\n')

  def addError(self, test, err):
    super().addError(test, err)
    logging.info('%s test-cases error occurredd , err %s', test, err)
    self.stream.write('\n' + Fore.YELLOW + 'ERROR' + Style.RESET_ALL + '\n')


class ColorfulTestRunner(unittest.TextTestRunner):
  resultclass = ColorfulTestResult
