import unittest

from tests.colorful_tests import ColorfulTestRunner
from helpers.cache.cache_factory import LocalCache
from helpers.cache.cache_factory import RedisCache
from helpers.cache.cache_helper import CacheKeyMaker


class CacheLocalTest(unittest.TestCase):

  @staticmethod
  def test_local_cache():
    # Initialize the cache
    cache = LocalCache(
        config={
            "local_base_dir": ".cache_local_data",
        },
        with_redis_client=RedisCache(config={
            "redis_host": "localhost",
            "redis_port": 1234,
            "redis_db": 0
        }).get_client(),
    )

    key_maker = CacheKeyMaker()

    key = key_maker.make_custom_key(
        custom_part=":".join(["ACR203", "FVS", "fire"]),
        managed_sample={
            "fire": "20%",
            "base": None
        },
    )

    # Set data in the cache
    cache.json_set(
        key,
        {"Fire": [2023, 2034, 3004]},
    )

    # Retrieve data from the cache
    result = cache.json_get(key)

    # Assert the retrieved data is correct
    expected_result = {"Fire": [2023, 2034, 3004]}
    assert (result == expected_result
           ), f"Expected {expected_result}, but got {result}"


if __name__ == "__main__":
  unittest.main(testRunner=ColorfulTestRunner())
