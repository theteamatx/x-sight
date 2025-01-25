"""Tests for the GCS Cache."""

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

  def test_gcs_cache(self):
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


if __name__ == "__main__":
  unittest.main(testRunner=ColorfulTestRunner())
