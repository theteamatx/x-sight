import json

import redis
from redis.commands.json.path import Path

from .cache_interface import CacheInterface


class RedisCache(CacheInterface):

  def __init__(self, config={}):
    try:
      self.redis_client = redis.StrictRedis(
          host=config.get("redis_host", "localhost"),
          port=config.get("redis_port", 6379),
          password=config.get("password", ""),
          db=config.get("redis_db", 0),
      )
      self.redis_client.ping()
    except (redis.ConnectionError, redis.TimeoutError) as e:
      print(f"RediConnection failed: {e}")
      self.redis_client = None

  def get_client(self):
    return self.redis_client

  def json_get(self, key):
    value = self.redis_client.json().get(key)
    return value if value else None

  def json_set(self, key, value):
    self.redis_client.json().set(key, Path.root_path(), value)
