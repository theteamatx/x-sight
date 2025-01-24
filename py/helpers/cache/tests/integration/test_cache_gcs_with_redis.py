"""Tests for GCS cache."""

import subprocess
import time
import unittest

from helpers.cache.cache_factory import GCSCache
from helpers.cache.cache_factory import RedisCache
import redis
from tests.colorful_tests import ColorfulTestRunner


class CacheGCSTest(unittest.TestCase):
  """Tests for GCS cache."""

  def wait_for_redis(self, host, port, timeout=30):
    """Waits for Redis to be ready.

    Args:
      host: The host of the Redis server.
      port: The port of the Redis server.
      timeout: The timeout in seconds.

    Raises:
      TimeoutError: If Redis is not ready within the timeout.
    """
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
    """Ends the Docker container.

    This end the docker container and delete all the keys in the redis cache.
    """
    if self.cache and self.cache.get_raw_redis_client():
      client = self.cache.get_raw_redis_client()
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

  def test_gcs_cache(self):
    """Tests the GCS cache."""

    self.cache = GCSCache(
        config={
            "gcs_base_dir": "test_sight_cache",
            "gcs_bucket": "cameltrain-sight",
        },
        with_redis_client=RedisCache(config={
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


if __name__ == "__main__":
  unittest.main(testRunner=ColorfulTestRunner())
