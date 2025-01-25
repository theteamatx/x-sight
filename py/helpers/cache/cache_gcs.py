"""GCS Cache.

This is GCS cache implementation. It uses Google Cloud Storage (GCS) to store
cached data. It also optionally uses Redis to improve performance and reduce
load on GCS.

The cache is configured with a GCS bucket name and a base directory within that
bucket. The base directory is used to separate different types of cached data.

"""

import json
import pathlib

from google.cloud import storage
from helpers.logs.logs_handler import logger as logging

from .cache_interface import CacheInterface
from .cache_redis import RedisCache

Path = pathlib.Path


class GCSCache(CacheInterface):
  """GCS Cache implementation."""

  def __init__(self, config=None, with_redis_cache: RedisCache | None = None):
    """Initializes the GCS Cache.

    Args:
      config: A dictionary of configuration options.
      with_redis_cache: A RedisCache client to use for caching.
    """
    config = config or {}
    gcs_client = storage.Client()
    bucket_name = config.get('gcs_bucket', 'cameltrain-sight')
    self.bucket = gcs_client.bucket(bucket_name=bucket_name)
    self.redis_cache = with_redis_cache
    self.gcs_base_dir = config.get('gcs_base_dir', 'sight_cache')

  def _gcs_cache_path(self, key: str, suffix: str = '.json'):
    """Returns the GCS cache path for the given key.

    Args:
      key: The key to get the cache path for.
      suffix: The suffix to add to the cache path.

    Returns:
      The GCS cache path.
    """
    return f'{self.gcs_base_dir}/{Path(key).with_suffix(suffix=suffix)}'

  def get_redis_client(self):
    """Returns the raw Redis client.

    Returns:
      The raw Redis client, or None if Redis is not enabled.
    """
    return (self.redis_cache and self.redis_cache.get_redis_client()) or None

  def json_get(self, key):
    """Gets a value from the cache.

    Args:
      key: The key to get the value for.

    Returns:
      The value from the cache, or None if not found.
    """
    if self.redis_cache and self.get_redis_client():
      try:
        value = self.redis_cache.json_get(key=key)
        if value:
          return value
      except Exception as e:  # pylint: disable=broad-exception-caught
        logging.warning('GOT THE ISSUE IN REDIS', e)
        return None
    blob = self.bucket.blob(self._gcs_cache_path(key=key.replace(':', '/')))
    if blob.exists():
      value = json.loads(blob.download_as_text())
      if self.redis_cache:
        self.redis_cache.json_set(key=key, value=value)
      return value
    return None

  def json_set(self, key, value):
    """Sets a value in the cache.

    Args:
      key: The key to set the value for.
      value: The value to set.
    """
    if self.redis_cache and self.get_redis_client():
      try:
        self.redis_cache.json_set(key=key, value=value)
      except Exception as e:  # pylint: disable=broad-exception-caught
        logging.warning('GOT THE ISSUE IN REDIS', e)
    blob = self.bucket.blob(self._gcs_cache_path(key=key.replace(':', '/')))
    blob.upload_from_string(json.dumps(value))

  def json_list_keys(self, prefix: str) -> list[str]:
    """Lists the keys in the cache.

    Args:
      prefix: The prefix to filter the keys by.

    Returns:
      A list of the keys in the cache.
    """
    if self.redis_cache and self.get_redis_client():
      try:
        return self.redis_cache.json_list_keys(prefix=prefix)
      except Exception as e:  # pylint: disable=broad-exception-caught
        logging.warning('GOT THE ISSUE IN REDIS', e)
    prefix = prefix.replace(':', '/')
    whole_prefix = self._gcs_cache_path(key=prefix, suffix='')
    blobs = self.bucket.list_blobs(prefix=whole_prefix)
    keys = []
    for blob in blobs:
      if blob.name.endswith('.json'):
        key = str(Path(blob.name).relative_to(
            self.gcs_base_dir)).split('.json')[0]
        keys.append(key)
    return keys
