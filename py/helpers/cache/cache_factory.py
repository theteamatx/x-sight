from .cache_gcs import GCSCache
from .cache_local import LocalCache
from .cache_redis import RedisCache
from .cache_none import NoneCache
from .cache_interface import CacheInterface


class CacheFactory:

    @staticmethod
    def get_cache(cache_type='local',
                  config={},
                  with_redis=None) -> CacheInterface | None:
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
            raise ValueError(f"Unknown cache type: {cache_type}")
