"""This module contains a Redis Cache implementation."""

import json
import pickle

from helpers.logs.logs_handler import logger as logging
from overrides import override
import redis
from redis.commands.json import path

from .cache_interface import CacheInterface

Path = path.Path

from typing import Any, List


class RedisCache(CacheInterface):
  """Redis Cache Implementation.

  This class implements the CacheInterface using Redis as the underlying
  technology. It provides methods for getting, setting, and listing keys in the
  cache.
  """

  def __init__(self, config=None):
    """Initializes the RedisCache object.

    Args:
      config: A dictionary containing configuration options for the Redis
        connection.
    """
    if config is None:
      config = {}
    try:
      self.redis_client = redis.StrictRedis(
          host=config.get("redis_host", "localhost"),
          port=config.get("redis_port", 1234),
          password=config.get("redis_pass", ""),
          db=config.get("redis_db", 0),
      )
      self.redis_client.ping()
    except (redis.ConnectionError, redis.TimeoutError) as e:
      logging.warning(f"RediConnection failed: {e}")
      self.redis_client = None
      raise e

  def get_redis_client(self):
    """Returns the raw Redis client object."""
    return self.redis_client or None

  def _is_redis_client_exist(self):
    """Raises an error if the Redis client is not connected."""
    if self.redis_client is None:
      logging.error("redis client not found..!!")
      raise ConnectionError("redis client not found , check connection !!")

  @override
  def get(self, key: str) -> Any:
    """Gets the value from the cache using key as string data"""
    self._is_redis_client_exist()
    string_value = self.redis_client.get(key)
    if string_value:
      return json.loads(string_value)
    return None

  @override
  def set(self, key: str, value: Any) -> None:
    """Set the key with value as string data"""
    self._is_redis_client_exist()
    try:
      string_value = json.dumps(value)
      self.redis_client.set(key, string_value)
    except TypeError as e:
      if "circular reference" in str(e):
        raise TypeError("Circular JSON object detected") from e

  @override
  def bin_get(self, key: str):
    """Gets a value from the cache using key as binary data"""
    self._is_redis_client_exist()
    value = self.redis_client.get(key)
    return pickle.loads(value) if value else None

  @override
  def bin_set(self, key, value):
    """Set the key with value as binary data"""
    self._is_redis_client_exist()
    self.redis_client.set(key, pickle.dumps(value))

  @override
  def json_get(self, key: str) -> Any:
    """Gets the JSON data from redis cache"""
    self._is_redis_client_exist()
    value = self.redis_client.json().get(key)
    return value if value else None

  @override
  def json_set(self, key: str, value: Any):
    """Sets the JSON data as key->value in redis cache"""
    self._is_redis_client_exist()
    self.redis_client.json().set(key, Path.root_path(), value)

  @override
  def list_keys(self, prefix: str) -> List[str]:
    """Lists all keys in the cache that match a given prefix.

    Args:
      prefix: The prefix to match keys against.

    Returns:
      A list of keys that match the prefix.
    """
    self._is_redis_client_exist()
    cursor = 0
    keys = []
    while True:
      cursor, batch_keys = self.redis_client.scan(cursor=cursor,
                                                  match=f"{prefix}*")
      keys.extend(batch_keys)
      if cursor == 0:
        break
    return [key.decode("utf-8") for key in keys]
