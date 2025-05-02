"""Cache Helper Module."""

import hashlib
import json
import pickle

import redis

from .cache_factory import RedisCache


def sort_nested_dict_or_list(
    data: dict[str, any] | list[any],) -> dict[str, any] | list[any]:
  """Sort the nested dict or list.

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
  def get_redis_instance(cache_type="none", config=None):
    if "with_redis" in cache_type:
      cache_redis: RedisCache = RedisCache(config)
      if cache_redis.get_redis_client() is None:
        raise ValueError(
            "Redis config maybe wrong or you haven't started the redis instance "
        )
      return cache_redis
    return None


class KeyMaker:
  """Cache key maker class."""

  def _serialize(self, *args, **kwargs):
    """Serialize arguments to create a unique key.

    This function serializes the provided positional and keyword arguments
    into a JSON string. The lists and dictionaries within the arguments are
    sorted recursively to ensure consistent serialization regardless of the
    order of elements.

    Args:
        *args: Positional arguments to be serialized.
        **kwargs: Keyword arguments to be serialized.

    Returns:
        str: A JSON string representing the serialized arguments.
    """
    combined = list(sort_nested_dict_or_list(list(args))) + list(
        sort_nested_dict_or_list(list(kwargs.values())))
    serialized_combined = json.dumps(
        sort_nested_dict_or_list(combined),
        separators=(",", ":"),
        ensure_ascii=True,
    )
    return serialized_combined

  def make_key(self, *args, **kwargs):
    """Generate a cache key with a specified part.

    This function constructs a cache key by hashing the serialized
    combination of the provided arguments.

    Args:
        *args: Positional arguments to be included in the key hash.
        **kwargs: Keyword arguments to be included in the key hash.

    Returns:
        str: The generated cache key in the format `hash(args+kwargs)`.

    Example:
        >>> key_maker = KeyMaker()
        >>> key_maker.make_key("arg1", key1="value1")
        'd14a021a6816261521263c1f86948807'
    """
    serialized = self._serialize(*args, **kwargs)
    key = f'{hashlib.md5(serialized.encode()).hexdigest()}'
    return key

  def make_custom_key(self, custom_part, *args, **kwargs):
    """Generate a custom cache key with a specified part.

    This function constructs a cache key by combining a custom part and a hash
    of the
    provided arguments. The key structure is
    `custom_part:hash(args+kwargs)`.

    Args:
        custom_part (str): The custom part string for the cache key.
        *args: Positional arguments to be included in the key hash.
        **kwargs: Keyword arguments to be included in the key hash.

    Returns:
        str: The generated custom cache key in the format
        `custom_part:hash(args+kwargs)`.

    Example:
        >>> key_maker = KeyMaker()
        >>> key_maker.make_custom_key("my_custom", "arg1", key1="value1")
        'my_custom:d14a021a6816261521263c1f86948807'
    """
    serialized = self._serialize(*args, **kwargs)
    hash_value = hashlib.md5(serialized.encode()).hexdigest()
    key = ":".join([custom_part, hash_value])
    return key

  def make_prefix_suffix_custom_key(self, prefix: str, suffix: str, *args,
                                    **kwargs):
    """Generate a custom cache key with prefix and suffix.

    This function constructs a cache key by combining a prefix, a hash of the
    provided arguments, and a suffix. The key structure is
    `prefix:hash(args+kwargs):suffix`.

    Args:
        prefix (str): The prefix string for the cache key.
        suffix (str): The suffix string for the cache key.
        *args: Positional arguments to be included in the key hash.
        **kwargs: Keyword arguments to be included in the key hash.

    Returns:
        str: The generated custom cache key in the format
        `prefix:hash(args+kwargs):suffix`.

    Example:
        >>> key_maker = KeyMaker()
        >>> key_maker.make_prefix_suffix_custom_key(
                "my_prefix", "my_suffix", "arg1", key1="value1"
            )
        'my_prefix:d14a021a6816261521263c1f86948807:my_suffix'

        >>> key_maker.make_prefix_suffix_custom_key(
                "another_prefix", "another_suffix", 123, key2="value2"
            )
        'another_prefix:74a28f5868796661256085a548c5689a:another_suffix'
    """
    serialized = self._serialize(*args, **kwargs)
    hash_value = hashlib.md5(serialized.encode()).hexdigest()
    key = ":".join([prefix, hash_value, suffix])
    return key


# * Currently below code is use for deleting the cache during the experiment
# * manually => a helper utility for developer only ignore the docstring
# * missing for them

# class CacheCleaner:
#   """Cache cleaner class."""

#   def __init__(self, host="localhost", port=1234, db=0, password=None):
#     self.redis_client = redis.Redis(
#         host=host, port=port, db=db, password=password
#     )

#   def delete_keys_with_prefix(self, prefix: str):
#     pattern = f"{prefix}*"
#     keys_to_delete = list(self.redis_client.scan_iter(match=pattern))

#     if keys_to_delete:
#       print(f"Deleting {len(keys_to_delete)} keys starting with '{prefix}'...")
#       self.redis_client.delete(*keys_to_delete)
#       print("Done.")
#     else:
#       print(f"No keys found with prefix '{prefix}'.")

#   def get_pickled_data(self, key: str):
#     data = self.redis_client.get(key)
#     if data is None:
#       print(f"No data found for key: {key}")
#       return None
#     try:
#       print(f"We got this data {data} => {pickle.loads(data)}")
#       return pickle.loads(data)
#     except pickle.UnpicklingError as e:
#       print(f"Error unpickling data for key '{key}': {e}")
#       return None

# if __name__ == "__main__":
#   cleaner = CacheCleaner(host="localhost", port=1234)
#   cleaner.delete_keys_with_prefix("sight_cache:*")
