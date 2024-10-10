import unittest

from helpers.cache.cache_factory import CacheFactory
from helpers.cache.cache_redis import RedisCache
from tests.colorful_tests import ColorfulTestRunner


class CacheRedisTest(unittest.TestCase):

  def test_redis_cache(self):
    # Configuration for the Redis cache
    config = {'redis_host': 'localhost', 'redis_port': 1234, 'redis_db': 0}

    # Initialize the Redis cache
    cache = RedisCache(config=config)

    self.assertIsNotNone(
        cache.get_client(),
        'Cache client is not found , check your redis connection !!')

    # Set data in the Redis cache
    cache.json_set("ACR203:2013:FVS:MANAGED:FIRE_0001011100",
                   {"Fire": [2023, 2034, 3004]})

    # Retrieve data from the Redis cache
    result = cache.json_get("ACR203:2013:FVS:MANAGED:FIRE_0001011100")

    # Assert the retrieved data is correct
    expected_result = {"Fire": [2023, 2034, 3004]}
    self.assertEqual(result, expected_result,
                     f"Expected {expected_result}, but got {result}")

  def test_redis_via_factory(self):

    client = CacheFactory.get_cache('redis', {
        'redis_host': 'localhost',
        'redis_port': 1234,
        'redis_db': 0
    })
    client.json_set('KEY', {'welcome': 'bhai'})
    self.assertTrue({'welcome': 'bhai'} == client.json_get('KEY'))
    # with self.assertRaises(Exception) as context:
    #      client.json_set('KEY', {})
    # self.assertTrue('redis client not found , check connection !!' in str(context.exception))


if __name__ == "__main__":
  unittest.main(testRunner=ColorfulTestRunner())
