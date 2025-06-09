"""Tests for the Local Cache."""

import json
import pathlib
import shutil
import unittest

from helpers.cache.cache_factory import LocalCache
from helpers.cache.cache_helper import KeyMaker
from tests.colorful_tests import ColorfulTestRunner

Path = pathlib.Path


class CacheLocalTest(unittest.TestCase):
  """Tests for the Local Cache."""

  def tearDown(self):
    super().tearDown()
    if Path(self.cache.base_dir).exists():
      shutil.rmtree(self.cache.base_dir)

  def setUp(self):
    super().setUp()
    # Initialize the cache
    self.cache = LocalCache(config={
        "local_base_dir": "/tmp/testing_dir",
    },)
    self.key_maker = KeyMaker()

  def test_cache_get_set(self):
    """Tests the Local Cache."""
    key = self.key_maker.make_custom_key(
        custom_part=":".join(["ACR203", "FVS", "fire"]),
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

  def test_cache_json_get_set(self):
    """Tests the Local Cache."""
    key = self.key_maker.make_custom_key(
        custom_part=":".join(["ACR203", "FVS", "fire"]),
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

  def test_list_keys(self):
    """Tests the list_keys method."""

    self.test_keys = [
        "logs:experiment1:chunk1",
        "logs:experiment1:chunk2",
        "logs:experiment2:chunk1",
    ]
    self.test_values = ["data1", "data2", "data3"]

    # Populate cache with test data
    for key, value in zip(self.test_keys, self.test_values):
      self.cache.set(key, value)

    # Test listing keys with prefix "logs:experiment1"
    keys = self.cache.list_keys("logs:experiment1")

    self.assertCountEqual(
        keys, ["logs:experiment1:chunk1", "logs:experiment1:chunk2"])

    # Test listing keys with prefix "logs"
    keys = self.cache.list_keys("logs")
    self.assertCountEqual(keys, self.test_keys)

    # Test listing keys with a non-existent prefix
    keys = self.cache.list_keys("non_existent_prefix")
    self.assertEqual(keys, [])


if __name__ == "__main__":
  unittest.main(testRunner=ColorfulTestRunner())
