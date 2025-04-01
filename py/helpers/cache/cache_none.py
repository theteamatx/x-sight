"""None cache implementation."""

from typing import Any

from helpers.logs.logs_handler import logger as logging

from .cache_interface import CacheInterface


class NoneCache(CacheInterface):
  """A cache implementation that does not cache anything."""

  def __init__(
      self,
      config: dict[str, Any] = None,
      with_redis_cache: Any = None,
  ):
    logging.warning('CACHE-TYPE-NONE -init # cache-ignore')

  def json_get(self, key: str) -> None:
    logging.warning('CACHE-TYPE-NONE -trying to get # cache-ignore')
    return None

  def json_set(self, key, value):
    logging.warning('CACHE-TYPE-NONE -trying to set # cache-ignore')

  def bin_get(self, key: str) -> None:
    logging.warning('CACHE-TYPE-NONE -trying to get # cache-ignore')
    return None

  def bin_set(self, key, value):
    logging.warning('CACHE-TYPE-NONE -trying to set # cache-ignore')

  def json_list_keys(self, prefix: str) -> list[str]:
    logging.warning('CACHE-TYPE-NONE -list keys # cache-ignore')
    return []
