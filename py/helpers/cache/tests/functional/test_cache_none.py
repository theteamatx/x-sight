"""Tests for the None cache."""

import unittest

from helpers.cache.cache_factory import CacheFactory
from helpers.cache.cache_helper import KeyMaker
from tests.colorful_tests import ColorfulTestRunner


class CacheNoneTest(unittest.TestCase):
  """Tests for the None cache."""

  def setUp(self):
    super().setUp()
    self.cache = CacheFactory.get_cache(cache_type="none")
    self.key_maker = KeyMaker()

  def test_json_list_keys(self):
    keys = self.cache.json_list_keys("any_prefix")
    self.assertEqual(keys, [])
    # Ensure no exceptions or errors occur

  def test_json_list_keys_empty_prefix(self):
    keys = self.cache.json_list_keys("")
    self.assertEqual(keys, [])
    # The result should always be an empty list

  def test_none_cache_json(self):
    """Tests the None cache."""

    key = self.key_maker.make_custom_key(
        custom_part=":".join(["ACR203", "FVS", "fire"]),
        managed_sample={
            "fire": "20%",
            "base": None
        },
    )

    # Set data in the cache
    self.cache.json_set(
        key,
        {"Fire": [2023, 2034, 3004]},
    )

    # Retrieve data from the cache
    result = self.cache.json_get(key)

    # Assert the retrieved data is correct
    expected_result = None
    assert (result == expected_result
           ), f"Expected {expected_result}, but got {result}"


if __name__ == "__main__":
  unittest.main(testRunner=ColorfulTestRunner())
