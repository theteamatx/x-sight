import json
import os
import logging
from .cache_interface import CacheInterface
from typing import Any


class NoneCache(CacheInterface):

    def __init__(self,
                 config: dict = {},
                 with_redis_client: Any | None = None):
        logging.warning('CACHE-TYPE-NONE -init # cache-ignore')

    def json_get(self, key: str) -> None:
        logging.warning('CACHE-TYPE-NONE -trying to get # cache-ignore')
        return None

    def json_set(self, key, value):
        logging.warning('CACHE-TYPE-NONE -trying to set # cache-ignore')
