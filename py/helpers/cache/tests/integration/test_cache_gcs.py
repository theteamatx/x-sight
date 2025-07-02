"""Tests for the GCS Cache."""

import json
import unittest

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
