"""Cache Helper Module."""

import hashlib
import json

from .cache_factory import CacheFactory
from .cache_factory import RedisCache


def sort_nested_dict_or_list(data: dict | list) -> dict | list:
  """Sort the nested dict or list

  Args:
      data (dict | list): input dict or list

  Returns:
      dict | list: sorted dict or list
  """
  if isinstance(data, dict):
    return {k: sort_nested_dict_or_list(v) for k, v in sorted(data.items())}
  elif isinstance(data, list):
    return [sort_nested_dict_or_list(x) for x in data]
  else:
    return data


class CacheConfig:
  """Cache configuration class."""

  @staticmethod
  def get_redis_instance(cache_type='none', config=None):
    if 'with_redis' in cache_type:
      cache_redis: RedisCache = RedisCache(config)
      if cache_redis.get_redis_client() is None:
        raise ValueError(
            "Redis config maybe wrong or you havn't started the redis instance "
        )
      return cache_redis
    return None


class CacheKeyMaker:
  """Cache key maker class."""

  def __init__(self):
    pass

  def _serialize(self, *args, **kwargs):
    """Serialize arguments to create a unique key."""
    combined = list(sort_nested_dict_or_list(list(args))) + list(
        sort_nested_dict_or_list(list(kwargs.values())))
    serialized_combined = json.dumps(
        sort_nested_dict_or_list(combined),
        separators=(',', ':'),
        ensure_ascii=True,
    )
    return serialized_combined

  def make_key(self, *args, **kwargs):
    """Generate a cache key based on provided arguments."""
    serialized = self._serialize(*args, **kwargs)
    key = f'{hashlib.md5(serialized.encode()).hexdigest()}'
    return key

  def make_custom_key(self, custom_part, *args, **kwargs):
    """Generate a custom cache key with a specified part."""
    serialized = self._serialize(*args, **kwargs)
    key = f'{custom_part}:{hashlib.md5(serialized.encode()).hexdigest()}'
    return key


# Currently below code is use for deleting the cache during the experiement manually => a helper utility for developer


class CacheCleaner:

  def __init__(self, host='localhost', port=1234, db=0, password=None):
    self.redis_client = redis.Redis(host=host,
                                    port=port,
                                    db=db,
                                    password=password)

  def delete_keys_with_prefix(self, prefix: str):
    pattern = f"{prefix}*"
    keys_to_delete = list(self.redis_client.scan_iter(match=pattern))

    if keys_to_delete:
      print(f"Deleting {len(keys_to_delete)} keys starting with '{prefix}'...")
      self.redis_client.delete(*keys_to_delete)
      print("Done.")
    else:
      print(f"No keys found with prefix '{prefix}'.")

  def get_pickled_data(self, key: str):
    data = self.redis_client.get(key)
    if data is None:
      print(f"No data found for key: {key}")
      return None
    try:
      print(f'We don this data {data} => {pickle.loads(data)}')
      return pickle.loads(data)
    except pickle.UnpicklingError as e:
      print(f"Error unpickling data for key '{key}': {e}")
      return None


if __name__ == "__main__":
  cleaner = CacheCleaner(host='localhost', port=1234)
  cleaner.delete_keys_with_prefix("sight_cache:*")
