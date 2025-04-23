"""Cache Constants Module."""


class CacheType():
  LOCAL = 'local'
  REDIS = 'redis'
  GCS = 'gcs'
  NONE = 'none'
  LOCAL_WITH_REDIS = 'local_with_redis'
  GCS_WITH_REDIS = 'gcs_with_redis'


class RedisConstants:
  """Class to hold Redis-related configuration constants."""

  REDIS_HOST = 'localhost'
  REDIS_PORT = 1234  # custom redis port
  REDIS_PASS = ''
  REDIS_DB = ''
