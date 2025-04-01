import json
import os
from pathlib import Path
import pickle

from helpers.logs.logs_handler import logger as logging
from redis import StrictRedis

from .cache_interface import CacheInterface
from .cache_redis import RedisCache


class LocalCache(CacheInterface):

  def __init__(self, config: dict = {}, with_redis_cache: RedisCache = None):
    self.base_dir = config.get("local_base_dir", "/tmp/.local_cache")
    self.redis_cache = with_redis_cache

  def _local_cache_path(self, key: str, suffix: str = ""):
    return Path(self.base_dir) / Path(key).with_suffix(suffix=suffix)

  def get_redis_client(self):
    return (self.redis_cache and self.redis_cache.get_redis_client()) or None

  def bin_get(self, key: str):
    if self.redis_cache and self.get_redis_client():
      try:
        value = self.redis_cache.bin_get(key=key)
        if value:
          return value
      except Exception as e:
        logging.warning("GOT THE ISSUE IN REDIS", e)
        return None
    path = self._local_cache_path(key.replace(":", "/"))
    if path.exists():
      with open(path, "rb") as file:
        value = pickle.load(file)
        if self.redis_cache and self.get_redis_client():
          self.redis_cache.bin_set(key, value)
        return value
    return None

  def bin_set(self, key, value):
    if self.redis_cache and self.get_redis_client():
      try:
        self.redis_cache.bin_set(key=key, value=value)
      except Exception as e:
        logging.warning("GOT THE ISSUE IN REDIS", e)
    path = self._local_cache_path(key.replace(":", "/"))
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as file:
      pickle.dump(value, file)

  def json_get(self, key: str):
    if self.redis_cache and self.get_redis_client():
      try:
        value = self.redis_cache.json_get(key=key)
        if value:
          return value
      except Exception as e:
        logging.warning("GOT THE ISSUE IN REDIS", e)
        return None
    path = self._local_cache_path(key.replace(":", "/"), suffix='.json')
    if path.exists():
      with open(path, "r") as file:
        value = json.load(file)
        if self.redis_cache and self.get_redis_client():
          self.redis_cache.json_set(key, value)
        return value
    return None

  def json_set(self, key, value):
    if self.redis_cache and self.get_redis_client():
      try:
        self.redis_cache.json_set(key=key, value=value)
      except Exception as e:
        logging.warning("GOT THE ISSUE IN REDIS", e)
    path = self._local_cache_path(key.replace(":", "/"), suffix='.json')
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as file:
      json.dump(value, file)

  def json_list_keys(self, prefix: str) -> list[str]:
    if self.redis_cache and self.get_redis_client():
      try:
        return self.redis_cache.json_list_keys(prefix=prefix)
      except Exception as e:
        logging.warning("GOT THE ISSUE IN REDIS", e)
    prefix = prefix.replace(':', '/')
    whole_prefix = self._local_cache_path(key=prefix, suffix='')
    if not whole_prefix.exists():
      return []
    return [
        str(file.relative_to(self.base_dir)).replace('/',
                                                     ':').replace('.json', '')
        for file in whole_prefix.rglob('*.json')
    ]
