import hashlib
import json


def sort_nested_dict(d):
  if isinstance(d, dict):
    return {k: sort_nested_dict(v) for k, v in sorted(d.items())}
  elif isinstance(d, list):
    return [sort_nested_dict(x) for x in d]
  else:
    return d

class CacheKeyMaker:

  def __init__(self):
    pass

  def _serialize(self, *args, **kwargs):
    """Serialize arguments to create a unique key"""
    combined = list(sorted(args)) + list(sort_nested_dict(kwargs.values()))
    serialized_combined = json.dumps(sort_nested_dict(combined), sort_keys=True)
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
