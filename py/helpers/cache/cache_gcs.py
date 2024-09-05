import json
from pathlib import Path

from google.cloud import storage
from redis import StrictRedis

from .cache_interface import CacheInterface


class GCSCache(CacheInterface):

  def __init__(self, config = {}, with_redis_client: StrictRedis | None = None):
    gcs_client = storage.Client()
    bucket_name = config.get('gcs_bucket', 'kokua-data')
    self.bucket = gcs_client.bucket(bucket_name=bucket_name)
    self.redis_client = with_redis_client
    self.gcs_base_dir = config.get("gcs_base_dir", "fvs_cache_data")

  def _gcs_cache_path(self, key: str):
    return f"{self.gcs_base_dir}/{Path(key).with_suffix('.json')}"

  def json_get(self, key):
    if self.redis_client:
      try:
        value = self.redis_client.json_get(key=key)
        if value:
          return value
      except Exception as e:
        print("GOT THE ISSUE IN REDIS", e)
        return None
    blob = self.bucket.blob(self._gcs_cache_path(key=key.replace(':', '/')))
    if blob.exists():
      return json.loads(blob.download_as_text())
    return None

  def json_set(self, key, value):
    if self.redis_client:
      try:
        self.redis_client.json_set(key=key, value=value)
      except Exception as e:
        print("GOT THE ISSUE IN REDIS", e)
    blob = self.bucket.blob(self._gcs_cache_path(key=key.replace(':', '/')))
    blob.upload_from_string(json.dumps(value))
