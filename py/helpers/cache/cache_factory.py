from .cache_gcs import GCSCache
from .cache_local import LocalCache
from .cache_redis import RedisCache


class CacheFactory:

  @staticmethod
  def get_cache(cache_type='local', config={}, with_redis=None):
    if cache_type == 'local':
      return LocalCache(config, with_redis)
    elif cache_type == 'gcs':
      return GCSCache(config, with_redis)
    elif cache_type == 'redis':
      return RedisCache(config)
    else:
      raise ValueError(f"Unknown cache type: {cache_type}")
