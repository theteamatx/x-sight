"""Cache Constants Module."""

import os


class CacheType():
  LOCAL = 'local'
  GCS = 'gcs'
  REDIS = 'redis'
  NONE = 'none'
  LOCAL_WITH_REDIS = 'local_with_redis'
  GCS_WITH_REDIS = 'gcs_with_redis'


class RedisConstants:
  """Class to hold Redis-related configuration constants."""

  REDIS_HOST = os.environ.get('REDIS_HOST', 'localhost')
  REDIS_PORT = os.environ.get('REDIS_PORT', '1234')
  REDIS_PASS = os.environ.get('REDIS_PASS', '')
  REDIS_DB = os.environ.get('REDIS_DB', '')
