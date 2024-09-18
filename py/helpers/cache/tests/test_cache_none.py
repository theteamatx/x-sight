import unittest

from tests.colorful_tests import ColorfulTestRunner
from helpers.cache.cache_factory import CacheFactory
from helpers.cache.cache_helper import CacheKeyMaker


class CacheNoneTest(unittest.TestCase):

    @staticmethod
    def test_none_cache():
        # Initialize the cache
        cache = CacheFactory.get_cache(cache_type='none')

        key_maker = CacheKeyMaker()

        key = key_maker.make_custom_key(
            custom_part=":".join(["ACR203", "FVS", "fire"]),
            managed_sample={
                "fire": "20%",
                "base": None
            },
        )

        # Set data in the cache
        cache.json_set(
            key,
            {"Fire": [2023, 2034, 3004]},
        )

        # Retrieve data from the cache
        result = cache.json_get(key)

        # Assert the retrieved data is correct
        expected_result = None
        assert (result == expected_result
                ), f"Expected {expected_result}, but got {result}"


if __name__ == "__main__":
    unittest.main(testRunner=ColorfulTestRunner())
