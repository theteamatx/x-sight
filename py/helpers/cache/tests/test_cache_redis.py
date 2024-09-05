import unittest

from fvs.tests.colorful_tests import ColorfulTestRunner
from helpers.cache.cache_redis import RedisCache


class CacheRedisTest(unittest.TestCase):

  def test_redis_cache(self):
    # Configuration for the Redis cache
    config = {'redis_host': 'localhost', 'redis_port': 1234, 'redis_db': 0}

    # Initialize the Redis cache
    cache = RedisCache(config=config)

    # Set data in the Redis cache
    cache.json_set("ACR203:2013:FVS:MANAGED:FIRE_0001011100",
                   {"Fire": [2023, 2034, 3004]})

    # Retrieve data from the Redis cache
    result = cache.json_get("ACR203:2013:FVS:MANAGED:FIRE_0001011100")

    # Assert the retrieved data is correct
    expected_result = {"Fire": [2023, 2034, 3004]}
    self.assertEqual(result, expected_result,
                     f"Expected {expected_result}, but got {result}")


if __name__ == "__main__":
  unittest.main(testRunner=ColorfulTestRunner())
