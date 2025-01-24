import unittest

from colorama import Fore
from colorama import init
from colorama import Style

# Initialize colorama
init(autoreset=True)


class ColorfulTestResult(unittest.TextTestResult):

  def addSuccess(self, test):
    super().addSuccess(test)
    print(test)
    self.stream.write('\n' + Fore.GREEN + 'PASS' + Style.RESET_ALL + '\n')

  def addFailure(self, test, err):
    super().addFailure(test, err)
    self.stream.write('\n' + Fore.RED + 'FAIL' + Style.RESET_ALL + '\n')

  def addError(self, test, err):
    super().addError(test, err)
    self.stream.write('\n' + Fore.YELLOW + 'ERROR' + Style.RESET_ALL + '\n')


class ColorfulTestRunner(unittest.TextTestRunner):
  resultclass = ColorfulTestResult
