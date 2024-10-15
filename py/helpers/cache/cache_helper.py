import hashlib
import json

from .cache_factory import CacheFactory


def sort_nested_dict_or_list(d):
  if isinstance(d, dict):
    return {k: sort_nested_dict_or_list(v) for k, v in sorted(d.items())}
  elif isinstance(d, list):
    return [sort_nested_dict_or_list(x) for x in d]
  else:
    return d


class CacheConfig:

  @staticmethod
  def get_redis_instance(cache_type='none'):
    if 'with_redis' in cache_type:
      cache_redis = CacheFactory.get_cache(cache_type='redis', config={})
      if cache_redis.get_client() is None:
        raise Exception(
            'Redis config maybe wrong or you havn\'t started the redis instance '
        )
      return cache_redis
    return None


# @staticmethod
# def get_cache_config(cache_type = 'none'):
#   match cache_type:
#     case'local':
#         return {

#         }


class CacheKeyMaker:

  def __init__(self):
    pass

  def _serialize(self, *args, **kwargs):
    """Serialize arguments to create a unique key"""
    combined = list(sort_nested_dict_or_list(list(args))) + list(
        sort_nested_dict_or_list(list(kwargs.values())))
    serialized_combined = json.dumps(sort_nested_dict_or_list(combined),
                                     sort_keys=True)
    return serialized_combined

  def make_key(self, *args, **kwargs):
    """Generate a cache key based on provided arguments"""
    serialized = self._serialize(*args, **kwargs)
    key = f"{hashlib.md5(serialized.encode()).hexdigest()}"
    return key

  def make_custom_key(self, custom_part, *args, **kwargs):
    """Generate a custom cache key with a specified part"""
    serialized = self._serialize(*args, **kwargs)
    key = f"{custom_part}:{hashlib.md5(serialized.encode()).hexdigest()}"
    return key
