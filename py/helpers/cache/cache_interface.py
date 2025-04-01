from abc import ABC
from abc import abstractmethod


class CacheInterface(ABC):

  @abstractmethod
  def bin_get(self, key: str):
    pass

  @abstractmethod
  def bin_set(self, key, value):
    pass

  @abstractmethod
  def json_get(self, key):
    pass

  @abstractmethod
  def json_set(self, key, value):
    pass

  @abstractmethod
  def json_list_keys(self, prefix: str) -> list[str]:
    """List all keys with a given prefix."""
    pass
