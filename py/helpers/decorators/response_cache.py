"""Flask API level cache decorator."""

import functools
import hashlib
import json
from typing import Optional

import flask
from helpers.cache.cache_factory import CacheFactory
from helpers.cache.cache_helper import sort_nested_dict_or_list

request = flask.request
jsonify = flask.jsonify
wraps = functools.wraps


def flask_api_level_cache(namespace="default",
                          cache_type="redis",
                          ttl: Optional[int] = None):
  """The flask api level cache decorator.

  It can easily add to endpoint to add the functionality of caching,
  it takes following params

  Args:
      namespace (str, optional): This is namespace which is part of hash key
        along with request payload to make the different key in-case of similar
        payload taking API endpoint. Defaults to "default".
      cache_type (str, optional): This is cache_type , this could be (REDIS ,
        GCS , LOCAL as per the need ). Defaults to "redis".
      ttl (Optional[int], optional): This is Time to Live which only applicable
        for redis cache type . Defaults to None.

  Returns:
      decorator: The decorator for the flask api level cache.
  """

  def decorator(func):

    @wraps(func)
    def wrapper(*args, **kwargs):
      raw_key = None
      cache_client = CacheFactory.get_cache(cache_type=cache_type)
      try:
        payload = request.get_json()
        raw_key = ":".join([
            "sight_cache",
            "apis_middleware",
            namespace,
            hashlib.md5(
                json.dumps(sort_nested_dict_or_list(
                    data=payload)).encode()).hexdigest(),
        ])
        cached = cache_client.get(key=raw_key)
        if cached:
          return jsonify(cached)
      except (
          json.JSONDecodeError,
          TypeError,
          KeyError,
      ) as e:
        print("Cache error (read):", e)

      # Run the function if cache miss
      result = func(*args, **kwargs)

      try:
        if raw_key:
          cache_client.set(key=raw_key, value=result.get_json())
          if cache_type == "redis" and ttl:
            cache_client.get_redis_client().expire(raw_key, ttl)
      except (
          json.JSONDecodeError,
          TypeError,
          KeyError,
      ) as e:
        print("Cache error (write):", e)

      return result

    return wrapper

  return decorator
