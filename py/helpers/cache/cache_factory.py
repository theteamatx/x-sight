"""A factory for creating different types of caches."""

from typing import Dict, Union

from .cache_gcs import GCSCache
from .cache_interface import CacheInterface
from .cache_local import LocalCache
from .cache_none import NoneCache
from .cache_redis import RedisCache
from .constants import CacheType
from .constants import RedisConstants


class CacheFactory:
  """A factory for creating different types of caches."""

  @staticmethod
  def get_cache(cache_type: Union[CacheType, str] = 'local',
                config: Union[Dict, None] = None,
                with_redis: Union[RedisCache, None] = None) -> CacheInterface:
    """Get a cache instance based on the given type.

    Args:
        cache_type (str): The type of cache to create.
        config (dict): A dictionary of configuration options.
        with_redis (None or RedisCache ): Whether to use Redis as the underlying cache.

    Returns:
        CacheInterface: An instance of the requested cache type.

    Raises:
        ValueError: If the cache type is unknown.
    """
    if with_redis == 'default':
      with_redis = RedisCache(
          config={
              "redis_host": RedisConstants.REDIS_HOST,
              "redis_port": RedisConstants.REDIS_PORT,
              "redis_pass": RedisConstants.REDIS_PASS
          })
    if config is None:
      config = {}
    if cache_type == CacheType.LOCAL_WITH_REDIS:
      return LocalCache(config, with_redis)
    elif cache_type == CacheType.GCS_WITH_REDIS:
      return GCSCache(config, with_redis)
    elif cache_type == CacheType.LOCAL:
      return LocalCache(config)
    elif cache_type == CacheType.GCS:
      return GCSCache(config)
    elif cache_type == CacheType.REDIS:
      if with_redis is not None:
        # we already have the redis cache object
        return with_redis
      return RedisCache(config)
    elif cache_type == CacheType.NONE:
      return NoneCache(config)
    else:
      raise ValueError(f'Unknown cache type: {cache_type}')
