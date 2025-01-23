import json
from pathlib import Path

from google.cloud import storage
from helpers.logs.logs_handler import logger as logging

from .cache_interface import CacheInterface
from .cache_redis import RedisCache


class GCSCache(CacheInterface):

  def __init__(self, config={}, with_redis_client: RedisCache | None = None):
    gcs_client = storage.Client()
    bucket_name = config.get('gcs_bucket', 'cameltrain-sight')
    self.bucket = gcs_client.bucket(bucket_name=bucket_name)
    self.redis_client = with_redis_client
    self.gcs_base_dir = config.get("gcs_base_dir", "sight_cache")

  def _gcs_cache_path(self, key: str, suffix: str = ".json"):
    return f"{self.gcs_base_dir}/{Path(key).with_suffix(suffix=suffix)}"

  def json_get(self, key):
    if self.redis_client:
      try:
        value = self.redis_client.json_get(key=key)
        if value:
          return value
      except Exception as e:
        logging.warning("GOT THE ISSUE IN REDIS", e)
        return None
    blob = self.bucket.blob(self._gcs_cache_path(key=key.replace(':', '/')))
    if blob.exists():
      value = json.loads(blob.download_as_text())
      if self.redis_client:
        self.redis_client.json_set(key=key, value=value)
      return value
    return None

  def json_set(self, key, value):
    if self.redis_client:
      try:
        self.redis_client.json_set(key=key, value=value)
      except Exception as e:
        logging.warning("GOT THE ISSUE IN REDIS", e)
    blob = self.bucket.blob(self._gcs_cache_path(key=key.replace(':', '/')))
    blob.upload_from_string(json.dumps(value))
