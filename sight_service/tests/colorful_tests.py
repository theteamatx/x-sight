import unittest

from colorama import Fore
from colorama import init
from colorama import Style
from helpers.logs.logs_handler import logger as logging

# Initialize colorama
init(autoreset=True)


class ColorfulTestResult(unittest.TextTestResult):

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
    logging.info('%s test-cases error occured , err %s', test, err)
    self.stream.write('\n' + Fore.YELLOW + 'ERROR' + Style.RESET_ALL + '\n')


class ColorfulTestRunner(unittest.TextTestRunner):
  resultclass = ColorfulTestResult
