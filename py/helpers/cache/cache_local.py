import json
import os
from pathlib import Path
import pickle
from typing import Any

from helpers.logs.logs_handler import logger as logging
from overrides import override
from redis import StrictRedis

from .cache_interface import CacheInterface
from .cache_redis import RedisCache


class LocalCache(CacheInterface):

  def __init__(self, config: dict = {}, with_redis_cache: RedisCache = None):
    """Initializes the Local Cache.

    Args:
      config: A dictionary of configuration options.
      with_redis_cache: A RedisCache client to use for caching.
    """
    self.base_dir = config.get("local_base_dir", "/tmp/.local_cache")
    self.redis_cache = with_redis_cache

  def _local_cache_path(self, key: str, suffix: str = ""):
    """Returns the local cache path for the given key"""
    return Path(self.base_dir) / Path(key.replace(
        ':', '/')).with_suffix(suffix=suffix)

  def get_redis_client(self):
    """Returns the Redis client."""
    return (self.redis_cache and self.redis_cache.get_redis_client()) or None

  def _get_from_redis(self, method, key):
    """Try to get value from redis and handle exceptions"""
    if self.get_redis_client():
      try:
        return getattr(self.redis_cache, method)(key)
      except Exception as e:
        logging.warning(f'Redis error in {method}: {e}')
    return None

  def _set_to_redis(self, method, key, value):
    """Try to set value in Redis and handle exceptions."""
    if self.get_redis_client():
      try:
        getattr(self.redis_cache, method)(key, value)
      except Exception as e:
        logging.warning(f"Redis error in {method}: {e}")

  @override
  def bin_get(self, key: str) -> Any:
    """Retrieve binary data from cache"""
    if (value := self._get_from_redis('bin_get', key)) is not None:
      return value
    path = self._local_cache_path(key)
    if path.exists():
      with open(path, "rb") as file:
        value = pickle.load(file)
        self._set_to_redis('bin_set', key, value)
        return value
    return None

  @override
  def bin_set(self, key: str, value: Any) -> None:
    """Store binary data in cache"""
    self._set_to_redis('bin_set', key, value)
    path = self._local_cache_path(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as file:
      pickle.dump(value, file)

  @override
  def json_get(self, key: str):
    """Retrieve JSON data from cache"""
    if (value := self._get_from_redis('json_get', key)) is not None:
      return value
    path = self._local_cache_path(key, suffix='.json')
    if path.exists():
      with open(path, "r") as file:
        value = json.load(file)
        self._set_to_redis('json_set', key, value)
        return value
    return None

  @override
  def json_set(self, key, value):
    """Store JSON data in cache"""
    self._set_to_redis('json_set', key, value)
    path = self._local_cache_path(key, suffix='.json')
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as file:
      json.dump(value, file)

  @override
  def json_list_keys(self, prefix: str) -> list[str]:
    """List all the keys with some prefix"""
    if (keys := self._get_from_redis('json_list_keys', prefix)) is not None:
      return keys
    prefix = prefix.replace(':', '/')
    whole_prefix = self._local_cache_path(key=prefix, suffix='')
    if not whole_prefix.exists():
      return []
    return [
        str(file.relative_to(self.base_dir)).replace('/',
                                                     ':').replace('.json', '')
        for file in whole_prefix.rglob('*.json')
    ]
