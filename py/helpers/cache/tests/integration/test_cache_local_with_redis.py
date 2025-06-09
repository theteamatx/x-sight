"""Tests for Local Cache with Redis."""

import json
import subprocess
import time
import unittest

from helpers.cache.cache_factory import LocalCache
from helpers.cache.cache_factory import RedisCache
from helpers.cache.cache_helper import KeyMaker
from helpers.cache.tests.integration.test_redis_contianer import (
    RedisContainerTest
)
import redis
from tests.colorful_tests import ColorfulTestRunner


class CacheLocalWithRedisTest(RedisContainerTest):
  """Tests for Local Cache with Redis."""

  def setUp(self):
    super().setUp()
    redis_client = self.__class__.redis_client
    redis_client.flushall()

  def test_local_get_set(self):
    """Tests the Local Cache."""
    self.key_maker = KeyMaker()
    self.cache = LocalCache(
        config={
            "local_base_dir": "/tmp/testing_dir",
        },
        with_redis_cache=RedisCache(config={
            "redis_host": "localhost",
            "redis_port": 1234,
            "redis_db": 0,
        }),
    )
    key = self.key_maker.make_custom_key(
        custom_part=":".join(["testing", "ACR203", "FVS", "fire"]),
        managed_sample={
            "fire": "20%",
            "base": None
        },
    )

    # Assert the retrieved data is correct
    expected_result = {"Fire": [2023, 2034, 3004]}

    # Set data in the cache
    self.cache.set(
        key,
        json.dumps(expected_result),
    )

    # Retrieve data from the cache
    result = self.cache.get(key)
    result = json.loads(result)
    assert (result == expected_result
           ), f"Expected {expected_result}, but got {result}"

  def test_local_json_get_set(self):
    """Tests the Local Cache."""
    self.key_maker = KeyMaker()
    self.cache = LocalCache(
        config={
            "local_base_dir": "/tmp/testing_dir",
        },
        with_redis_cache=RedisCache(config={
            "redis_host": "localhost",
            "redis_port": 1234,
            "redis_db": 0,
        }),
    )
    key = self.key_maker.make_custom_key(
        custom_part=":".join(["testing", "ACR203", "FVS", "fire"]),
        managed_sample={
            "fire": "20%",
            "base": None
        },
    )

    # Assert the retrieved data is correct
    expected_result = {"Fire": [2023, 2034, 3004]}

    # Set data in the cache
    self.cache.json_set(
        key,
        expected_result,
    )

    # Retrieve data from the cache
    result = self.cache.json_get(key)

    assert (result == expected_result
           ), f"Expected {expected_result}, but got {result}"

  def test_local_bin_get_set(self):
    """Tests the Local Cache."""
    self.key_maker = KeyMaker()
    self.cache = LocalCache(
        config={
            "local_base_dir": "/tmp/testing_dir",
        },
        with_redis_cache=RedisCache(config={
            "redis_host": "localhost",
            "redis_port": 1234,
            "redis_db": 0,
        }),
    )
    key = self.key_maker.make_custom_key(
        custom_part=":".join(["testing", "ACR203", "FVS", "fire"]),
        managed_sample={
            "fire": "20%",
            "base": None
        },
    )

    # Assert the retrieved data is correct
    expected_result = {"Fire": [2023, 2034, 3004]}

    # Set data in the cache
    self.cache.bin_set(
        key,
        expected_result,
    )

    # Retrieve data from the cache
    result = self.cache.bin_get(key)

    assert (result == expected_result
           ), f"Expected {expected_result}, but got {result}"

  def test_list_keys(self):
    """Tests the list_keys method."""
    self.cache = LocalCache(
        config={
            "local_base_dir": "/tmp/testing_dir",
        },
        with_redis_cache=RedisCache(config={
            "redis_host": "localhost",
            "redis_port": 1234,
            "redis_db": 0,
        }),
    )
    self.test_keys = [
        "testing:logs:experiment1:chunk1",
        "testing:logs:experiment1:chunk2",
        "testing:logs:experiment2:chunk1",
    ]
    self.test_values = ["data1", "data2", "data3"]

    # Populate cache with test data
    for key, value in zip(self.test_keys, self.test_values):
      self.cache.json_set(key, value)

    # Test listing keys with prefix "logs:experiment1"
    keys = self.cache.list_keys("testing:logs:experiment1")

    self.assertEqual(
        sorted(keys, key=json.dumps),
        sorted(
            [
                "testing:logs:experiment1:chunk1",
                "testing:logs:experiment1:chunk2",
            ],
            key=json.dumps,
        ),
    )

    # Test listing keys with prefix "logs"
    keys = self.cache.list_keys("testing:logs")
    self.assertCountEqual(keys, self.test_keys)

    # Test listing keys with a non-existent prefix
    keys = self.cache.list_keys("non_existent_prefix")
    self.assertEqual(keys, [])


if __name__ == "__main__":
  unittest.main(testRunner=ColorfulTestRunner())
