import json

from helpers.logs.logs_handler import logger as logging
import redis
from redis.commands.json.path import Path

from .cache_interface import CacheInterface


class RedisCache(CacheInterface):

  def __init__(self, config={}):
    try:
      self.raw_redis_client = redis.StrictRedis(
          host=config.get("redis_host", "localhost"),
          port=config.get("redis_port", 1234),
          password=config.get("password", ""),
          db=config.get("redis_db", 0),
      )
      self.raw_redis_client.ping()
    except (redis.ConnectionError, redis.TimeoutError) as e:
      logging.warning(f"RediConnection failed: {e}")
      self.redis_client = None
      raise e

  def get_raw_redis_client(self):
    return self.raw_redis_client or None

  def _is_redis_client_exist(self):
    if self.raw_redis_client is None:
      logging.error('redis client not found..!!')
      raise Exception("redis client not found , check connection !!")

  def json_get(self, key):
    self._is_redis_client_exist()
    value = self.raw_redis_client.json().get(key)
    return value if value else None

  def json_set(self, key, value):
    self._is_redis_client_exist()
    self.raw_redis_client.json().set(key, Path.root_path(), value)

  def json_list_keys(self, prefix: str) -> list[str]:
    self._is_redis_client_exist()
    cursor = 0
    keys = []
    while True:
      cursor, batch_keys = self.raw_redis_client.scan(cursor=cursor,
                                                      match=f"{prefix}*")
      keys.extend(batch_keys)
      if cursor == 0:
        break
    return [key.decode('utf-8') for key in keys]
