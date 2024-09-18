import unittest

from tests.colorful_tests import ColorfulTestRunner
from helpers.cache.cache_factory import GCSCache
from helpers.cache.cache_factory import RedisCache


class CacheGCSTest(unittest.TestCase):

  @staticmethod
  def test_gcs_cache():
    # Initialize the cache
    cache = GCSCache(
        config={
            "gcs_base_dir": "test_sight_cache",
            "gcs_bucket": "cameltrain-sight"
        },
        with_redis_client=RedisCache(config={
            "redis_host": "localhost",
            "redis_port": 1234,
            "redis_db": 0
        }).get_client(),
    )

    # Set data in the cache
    cache.json_set(
        "ACR203:2013:FVS:MANAGED:FIRE_0001011100",
        {"Fire": [2023, 2034, 3004, "Nice And Working"]},
    )

    # Retrieve data from the cache
    result = cache.json_get("ACR203:2013:FVS:MANAGED:FIRE_0001011100")

    # Assert the retrieved data is correct
    expected_result = {"Fire": [2023, 2034, 3004, "Nice And Working"]}
    assert (result == expected_result
           ), f"Expected {expected_result}, but got {result}"


if __name__ == "__main__":
  unittest.main(testRunner=ColorfulTestRunner())
