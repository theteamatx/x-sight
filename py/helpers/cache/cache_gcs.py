"""GCS Cache.

This is GCS cache implementation. It uses Google Cloud Storage (GCS) to store
cached data. It also optionally uses Redis to improve performance and reduce
load on GCS.

The cache is configured with a GCS bucket name and a base directory within that
bucket. The base directory is used to separate different types of cached data.

"""

import hashlib
import json
import pathlib
import pickle
import threading

from google.cloud import storage
from helpers.logs.logs_handler import logger as logging

from .cache_interface import CacheInterface
from .cache_redis import RedisCache

Path = pathlib.Path
from typing import Any, List

from overrides import override


class GCSCache(CacheInterface):
  """GCS Cache implementation."""

  _instances = {}
  _lock = threading.Lock()  # to make GCSCache thread-safe

  def __new__(cls, config=None):
    config = config or {}
    # The GCSCache instance is unique solely based on 'config'.
    # 'with_redis_cache' is used only during the first __init__ call for a given config.
    config_key = cls._get_config_hash(config)
    with cls._lock:
      if config_key not in cls._instances:
        instance = super(GCSCache, cls).__new__(cls)
        cls._instances[config_key] = instance
        instance._initialized = False  # so __init__ only runs once per instance
      return cls._instances[config_key]

  @staticmethod
  def _get_config_hash(config: dict) -> str:
    config_str = json.dumps(config, sort_keys=True)
    return hashlib.sha256(config_str.encode()).hexdigest()

  def __init__(self, config=None, with_redis_cache: RedisCache = None):
    """Initializes the GCS Cache.

    Args:
      config: A dictionary of configuration options.
      with_redis_cache: A RedisCache client to use for caching.
    """
    if getattr(self, '_initialized', False):
      return  # prevent re-init

    config = config or {}
    gcs_client = storage.Client()
    bucket_name = config.get('gcs_bucket', 'cameltrain-sight')
    self.bucket = gcs_client.bucket(bucket_name=bucket_name)
    self.redis_cache = with_redis_cache
    self.gcs_base_dir = config.get('gcs_base_dir', 'sight_object_storage_cache')
    self._initialized = True

  def _gcs_cache_path(self, key: str, suffix: str = '.json'):
    """Returns the GCS cache path for the given key"""
    return str(
        Path(self.gcs_base_dir) /
        Path(key.replace(':', '/')).with_suffix(suffix=suffix))

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
  def get(self, key: str):
    """Retrieve data from cache"""
    if (value := self._get_from_redis('get', key)) is not None:
      return value
    blob = self.bucket.blob(self._gcs_cache_path(key=key))
    logging.debug('This is our path for key %s', blob.name)
    if blob.exists():
      value = blob.download_as_text()
      self._set_to_redis('set', key, value)
      return value
    return None

  @override
  def set(self, key: str, value: Any):
    """Store data in cache"""
    self._set_to_redis('set', key, value)
    blob = self.bucket.blob(self._gcs_cache_path(key=key))
    logging.info('This is our path for key %s', blob.name)
    blob.upload_from_string(value)

  @override
  def bin_get(self, key: str) -> Any:
    """Retrieve binary data from cache"""
    if (value := self._get_from_redis('bin_get', key)) is not None:
      return value
    blob = self.bucket.blob(self._gcs_cache_path(key=key))
    if blob.exists():
      value = pickle.loads(blob.download_as_bytes())
      self._set_to_redis('bin_set', key, value)
      return value
    return None

  @override
  def bin_set(self, key: str, value: Any) -> None:
    """Store binary data in cache"""
    self._set_to_redis('bin_set', key, value)
    blob = self.bucket.blob(self._gcs_cache_path(key=key))
    blob.upload_from_string(pickle.dumps(value))

  @override
  def json_get(self, key: str) -> Any:
    """Retrieve JSON data from cache"""
    if (value := self._get_from_redis('json_get', key)) is not None:
      return value
    blob = self.bucket.blob(self._gcs_cache_path(key=key))
    if blob.exists():
      value = json.loads(blob.download_as_text())
      self._set_to_redis('json_set', key, value)
      return value
    return None

  @override
  def json_set(self, key: str, value: Any) -> None:
    """Store JSON data in cache"""
    self._set_to_redis('json_set', key, value)
    blob = self.bucket.blob(self._gcs_cache_path(key=key))
    blob.upload_from_string(json.dumps(value))

  @override
  def list_keys(self, prefix: str) -> List[str]:
    """List all the keys with some prefix"""
    if (keys := self._get_from_redis('list_keys', prefix)) is not None:
      return keys
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
