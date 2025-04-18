"""Interface for log storage strategies."""

import abc

abstractmethod = abc.abstractmethod
ABC = abc.ABC


class ILogStorageCollectStrategy(ABC):

  @abstractmethod
  def save_logs(self, logs):
    pass

  @abstractmethod
  def collect_logs(self) -> list[str]:
    pass
