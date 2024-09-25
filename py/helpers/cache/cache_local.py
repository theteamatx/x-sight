import json
import os
from helpers.logs.logs_handler import logger as logging
from pathlib import Path

from redis import StrictRedis

from .cache_interface import CacheInterface


class LocalCache(CacheInterface):

    def __init__(self,
                 config: dict = {},
                 with_redis_client: StrictRedis | None = None):
        base_dir = config.get("local_base_dir", "./.cache_local_data")
        self.redis_client = with_redis_client
        self.current_script_path = os.path.dirname(os.path.abspath(__file__))
        self.base_dir = os.path.join(self.current_script_path,
                                     f"../../{base_dir}")

    def _local_cache_path(self, key: str):
        return Path(self.base_dir) / Path(key).with_suffix(".json")

    def json_get(self, key: str):
        if self.redis_client:
            try:
                value = self.redis_client.json_get(key=key)
                if value:
                    return value
            except Exception as e:
                logging.warning("GOT THE ISSUE IN REDIS", e)
                return None
        path = self._local_cache_path(key.replace(":", "/"))
        if path.exists():
            with open(path, "r") as file:
                value = json.load(file)
                if self.redis_client:
                    self.redis_client.json_set(key, value)
                return value
        return None

    def json_set(self, key, value):
        if self.redis_client:
            try:
                self.redis_client.json_set(key=key, value=value)
            except Exception as e:
                logging.warning("GOT THE ISSUE IN REDIS", e)
        path = self._local_cache_path(key.replace(":", "/"))
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as file:
            json.dump(value, file)
