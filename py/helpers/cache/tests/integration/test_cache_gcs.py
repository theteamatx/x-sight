"""Tests for the GCS Cache."""

import json
import time
import unittest
import uuid

from helpers.cache.cache_factory import GCSCache
from tests.colorful_tests import ColorfulTestRunner


class CacheGCSTest(unittest.TestCase):
  """Tests for the GCS Cache."""

  def setUp(self):
    super().setUp()
    # Initialize the cache
    self.cache = GCSCache(config={
        "gcs_base_dir": "test_sight_cache",
        "gcs_bucket": "cameltrain-sight",
    },)

  def test_singleton_with_same_config(self):
    config = {
        "gcs_bucket": "my-test-cache-bucket",  # Replace with real test bucket
        "gcs_base_dir": f"singleton_test/{uuid.uuid4()}",
    }

    gcs1 = GCSCache(config)
    gcs2 = GCSCache(config)

    self.assertIs(
        gcs1,
        gcs2,
        "Two instances with the same config should return the same singleton"
        " instance.",
    )

  def test_different_config_returns_different_instance(self):
    config1 = {
        "gcs_bucket": "my-test-cache-bucket",  # Replace with real test bucket
        "gcs_base_dir": f"singleton_test_a/{uuid.uuid4()}",
    }
    config2 = {
        "gcs_bucket": "my-test-cache-bucket",  # Same bucket but different dir
        "gcs_base_dir": f"singleton_test_b/{uuid.uuid4()}",
    }

    gcs1 = GCSCache(config1)
    gcs2 = GCSCache(config2)

    self.assertIsNot(
        gcs1,
        gcs2,
        "Two instances with different config should return different"
        " instances.",
    )

  def test_singleton_is_faster_with_same_config(self):
    base_dir = f"singleton_perf_test/{uuid.uuid4()}"
    config = {
        "gcs_bucket": 'test_sight_cache',
        "gcs_base_dir": base_dir,
    }

    # First instantiation (cold start)
    start = time.perf_counter()
    _ = GCSCache(config)
    cold_time = time.perf_counter() - start

    # Multiple subsequent calls with same config
    start = time.perf_counter()
    for _ in range(1000):
      _ = GCSCache(config)
    warm_time = time.perf_counter() - start

    print(f"Cold init time: {cold_time:.6f}s")
    print(f"Warm repeated init time (1000x): {warm_time:.6f}s")

    self.assertLess(
        warm_time, cold_time * 5,
        "Subsequent instantiations with same config should be significantly faster than cold start"
    )

  def test_gcs_get_set(self):
    """Tests the GCS Cache."""
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
    """Tests the GCS Cache."""
    # Set data in the cache
    self.cache.json_set(
        "ACR203:2013:FVS:MANAGED:FIRE_0001011100",
        {"Fire": [2023, 2034, 3004, "Nice And Working"]},
    )

    # Retrieve data from the cache
    result = self.cache.json_get("ACR203:2013:FVS:MANAGED:FIRE_0001011100")

    # Assert the retrieved data is correct
    expected_result = {"Fire": [2023, 2034, 3004, "Nice And Working"]}
    assert (result == expected_result
           ), f"Expected {expected_result}, but got {result}"

  def test_gcs_bin_get_set(self):
    """Tests the GCS Cache."""
    # Set data in the cache
    self.cache.bin_set(
        "ACR203:2013:FVS:MANAGED:FIRE_0001011100",
        {"Fire": [2023, 2034, 3004, "Nice And Working"]},
    )

    # Retrieve data from the cache
    result = self.cache.bin_get("ACR203:2013:FVS:MANAGED:FIRE_0001011100")

    # Assert the retrieved data is correct
    expected_result = {"Fire": [2023, 2034, 3004, "Nice And Working"]}
    assert (result == expected_result
           ), f"Expected {expected_result}, but got {result}"


if __name__ == "__main__":
  unittest.main(testRunner=ColorfulTestRunner())
