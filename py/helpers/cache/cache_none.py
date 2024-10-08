import json
import os
from typing import Any

from helpers.logs.logs_handler import logger as logging

from .cache_interface import CacheInterface


class NoneCache(CacheInterface):

  def __init__(self, config: dict = {}, with_redis_client: Any | None = None):
    logging.warning('CACHE-TYPE-NONE -init # cache-ignore')

  def json_get(self, key: str) -> None:
    logging.warning('CACHE-TYPE-NONE -trying to get # cache-ignore')
    return None

  def json_set(self, key, value):
    logging.warning('CACHE-TYPE-NONE -trying to set # cache-ignore')
