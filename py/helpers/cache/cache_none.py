"""None cache implementation."""

from typing import Any

from helpers.logs.logs_handler import logger as logging
from overrides import override

from .cache_interface import CacheInterface


class NoneCache(CacheInterface):
  """A cache implementation that does not cache anything."""

  def __init__(
      self,
      config: dict[str, Any] = None,
      with_redis_cache: Any = None,
  ):
    logging.warning('CACHE-TYPE-NONE -init # cache-ignore')

  @override
  def json_get(self, key: str) -> None:
    logging.warning('CACHE-TYPE-NONE -trying to json get # cache-ignore')
    return None

  @override
  def json_set(self, key, value):
    logging.warning('CACHE-TYPE-NONE -trying to json set # cache-ignore')

  @override
  def bin_get(self, key: str) -> None:
    logging.warning('CACHE-TYPE-NONE -trying to bin get # cache-ignore')
    return None

  @override
  def bin_set(self, key, value):
    logging.warning('CACHE-TYPE-NONE -trying to bin set # cache-ignore')

  @override
  def json_list_keys(self, prefix: str) -> list[str]:
    logging.warning('CACHE-TYPE-NONE -list keys # cache-ignore')
    return []
