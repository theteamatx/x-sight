"""Tests for GCS cache."""

import json
import subprocess
import time
import unittest

from helpers.cache.cache_factory import GCSCache
from helpers.cache.cache_factory import RedisCache
from helpers.cache.tests.integration.test_redis_contianer import (
    RedisContainerTest
)
from helpers.logs.logs_handler import logger as logging
import redis
from tests.colorful_tests import ColorfulTestRunner


class CacheGCSTest(RedisContainerTest):
  """Tests for GCS cache."""

  def setUp(self):
    super().setUp()
    redis_client = self.__class__.redis_client
    redis_client.flushall()

  def test_gcs_get_set(self):
    """Tests the GCS Cache."""

    self.cache = GCSCache(
        config={
            "gcs_base_dir": "test_sight_cache",
            "gcs_bucket": "cameltrain-sight",
        },
        with_redis_cache=RedisCache(config={
            "redis_host": "localhost",
            "redis_port": 1234,
            "redis_db": 0,
        }),
    )

    # Set data in the cache
    self.cache.set(
        "ACR203:2013:FVS:MANAGED:FIRE_0001011100",
        json.dumps({"Fire": [2023, 2034, 3004, "Nice And Working"]}),
    )

    # Retrieve data from the cache
    result = self.cache.get("ACR203:2013:FVS:MANAGED:FIRE_0001011100")
    result = json.loads(result)
    # Assert the retrieved data is correct
    expected_result = {"Fire": [2023, 2034, 3004, "Nice And Working"]}
    assert (result == expected_result
           ), f"Expected {expected_result}, but got {result}"

  def test_gcs_json_get_set(self):
    """Tests the GCS cache."""

    self.cache = GCSCache(
        config={
            "gcs_base_dir": "test_sight_cache",
            "gcs_bucket": "cameltrain-sight",
        },
        with_redis_cache=RedisCache(config={
            "redis_host": "localhost",
            "redis_port": 1234,
            "redis_db": 0,
        }),
    )
    # Set data in the cache
    self.cache.json_set(
        "testing:ACR203:2013:FVS:MANAGED:FIRE_0001011100",
        {"Fire": [2023, 2034, 3004, "Nice And Working"]},
    )

    # Retrieve data from the self.cache
    result = self.cache.json_get(
        "testing:ACR203:2013:FVS:MANAGED:FIRE_0001011100")

    # Assert the retrieved data is correct
    expected_result = {"Fire": [2023, 2034, 3004, "Nice And Working"]}
    assert (result == expected_result
           ), f"Expected {expected_result}, but got {result}"

  def test_gcs_bin_get_set(self):
    """Tests the GCS cache."""

    self.cache = GCSCache(
        config={
            "gcs_base_dir": "test_sight_cache",
            "gcs_bucket": "cameltrain-sight",
        },
        with_redis_cache=RedisCache(config={
            "redis_host": "localhost",
            "redis_port": 1234,
            "redis_db": 0,
        }),
    )
    # Set data in the cache
    self.cache.bin_set(
        "testing:ACR203:2013:FVS:MANAGED:FIRE_0001011100",
        {"Fire": [2023, 2034, 3004, "Nice And Working"]},
    )

    # Retrieve data from the self.cache
    result = self.cache.bin_get(
        "testing:ACR203:2013:FVS:MANAGED:FIRE_0001011100")

    # Assert the retrieved data is correct
    expected_result = {"Fire": [2023, 2034, 3004, "Nice And Working"]}
    assert (result == expected_result
           ), f"Expected {expected_result}, but got {result}"


if __name__ == "__main__":
  unittest.main(testRunner=ColorfulTestRunner())
