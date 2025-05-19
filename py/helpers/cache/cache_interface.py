from abc import ABC
from abc import abstractmethod
from typing import Any, List


class CacheInterface(ABC):

  @abstractmethod
  def get(self, key: str) -> Any:
    """Get the data from cache"""
    pass

  @abstractmethod
  def set(self, key: str, value: Any) -> None:
    """Set the data to the cache"""
    pass

  @abstractmethod
  def bin_get(self, key: str) -> Any:
    """Get the Binary data from cache"""
    pass

  @abstractmethod
  def bin_set(self, key: str, value: Any) -> None:
    """Set the Binary Data to the cache"""
    pass

  @abstractmethod
  def json_get(self, key: str) -> Any:
    """Get the JSON data from the cache"""
    pass

  @abstractmethod
  def json_set(self, key: str, value: Any) -> None:
    """Set the JSON data to the cache"""
    pass

  @abstractmethod
  def list_keys(self, prefix: str) -> List[str]:
    """List all keys with a given prefix."""
    pass
