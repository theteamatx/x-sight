"""A factory for creating different types of caches."""

from .cache_gcs import GCSCache
from .cache_interface import CacheInterface
from .cache_local import LocalCache
from .cache_none import NoneCache
from .cache_redis import RedisCache


class CacheFactory:
  """A factory for creating different types of caches."""

  @staticmethod
  def get_cache(cache_type='local',
                config=None,
                with_redis=None) -> CacheInterface:
    """Get a cache instance based on the given type.

    Args:
        cache_type (str): The type of cache to create.
        config (dict): A dictionary of configuration options.
        with_redis (bool): Whether to use Redis as the underlying cache.

    Returns:
        CacheInterface: An instance of the requested cache type.

    Raises:
        ValueError: If the cache type is unknown.
    """
    if config is None:
      config = {}
    if cache_type == 'local_with_redis':
      return LocalCache(config, with_redis)
    elif cache_type == 'gcs_with_redis':
      return GCSCache(config, with_redis)
    elif cache_type == 'local':
      return LocalCache(config)
    elif cache_type == 'gcs':
      return GCSCache(config)
    elif cache_type == 'redis':
      return RedisCache(config)
    elif cache_type == 'none':
      return NoneCache(config)
    else:
      raise ValueError(f'Unknown cache type: {cache_type}')
