"""Tests for Local Cache with Redis."""

import json
import subprocess
import time
import unittest

from helpers.cache.cache_factory import LocalCache
from helpers.cache.cache_factory import RedisCache
from helpers.cache.cache_helper import CacheKeyMaker
import redis
from tests.colorful_tests import ColorfulTestRunner


class CacheLocalWithRedisTest(unittest.TestCase):
  """Tests for Local Cache with Redis."""

  def wait_for_redis(self, host, port, timeout=30):
    start_time = time.time()
    while time.time() - start_time < timeout:
      try:
        client = redis.StrictRedis(host=host, port=port)
        client.ping()
        print("Redis is ready!")
        return
      except redis.ConnectionError:
        time.sleep(1)
    raise TimeoutError(f"Redis not ready after {timeout} seconds")

  def _end_container(self):
    """Stops the Docker containers."""
    if self.cache and self.cache.get_redis_client():
      client = self.cache.get_redis_client()
      keys_to_delete = client.keys("testing:*")
      if keys_to_delete:
        client.delete(*keys_to_delete)  # Delete all matching keys
    try:
      print("Stopping Docker containers ...")
      subprocess.run(["docker-compose", "down"], check=True)
      print("Docker containers stopped successfully...")
    except subprocess.CalledProcessError as e:
      print(f"Failed to stop Docker containers : {e}")
      raise e

  def tearDown(self):
    super().tearDown()
    self._end_container()

  def setUp(self):
    super().setUp()
    self.cache = None
    self._end_container()
    try:
      print("Starting Docker containers ...")
      subprocess.run(["docker-compose", "up", "-d"], check=True)
      # Wait for Redis to be ready
      self.wait_for_redis("localhost", 1234)
      print("Docker containers started successfully...")
    except subprocess.CalledProcessError as e:
      print(f"Failed to start Docker containers : {e}")
      raise e

  def test_local_cache(self):
    """Tests the Local Cache."""
    self.key_maker = CacheKeyMaker()
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

  def test_json_list_keys(self):
    """Tests the json_list_keys method."""
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
    keys = self.cache.json_list_keys("testing:logs:experiment1")

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
    keys = self.cache.json_list_keys("testing:logs")
    self.assertCountEqual(keys, self.test_keys)

    # Test listing keys with a non-existent prefix
    keys = self.cache.json_list_keys("non_existent_prefix")
    self.assertEqual(keys, [])


if __name__ == "__main__":
  unittest.main(testRunner=ColorfulTestRunner())
