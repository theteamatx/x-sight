"""Tests for the Redis cache."""

import json
import time
import unittest

from helpers.cache.cache_factory import CacheFactory
from helpers.cache.cache_redis import RedisCache
from helpers.cache.tests.integration.test_redis_contianer import (
    RedisContainerTest
)
from helpers.logs.logs_handler import logger as logging
import redis
from tests.colorful_tests import ColorfulTestRunner


class CacheRedisTest(RedisContainerTest):
  """Tests for the Redis cache."""

  @classmethod
  def setUpClass(cls):
    logging.info('Setting up the Container !!')
    super().setUpClass()

  @classmethod
  def tearDownClass(cls):
    logging.info('Tearing Down the Container !!')
    super().tearDownClass()

  def setUp(self):
    redis_client = self.__class__.redis_client
    redis_client.flushall()

  def test_redis_get_set(self):
    """Tests the Redis cache."""

    # Configuration for the Redis cache
    config = {'redis_host': 'localhost', 'redis_port': 1234, 'redis_db': 0}
    # Initialize the Redis cache
    self.cache = RedisCache(config=config)

    self.assertIsNotNone(
        self.cache.get_redis_client(),
        'Cache client is not found , check your redis connection !!',
    )

    # Set data in the Redis cache
    self.cache.set('testing:test1:0', json.dumps({'Fire': [2023, 2034, 3004]}))

    # Retrieve data from the Redis cache
    result = self.cache.get('testing:test1:0')
    result = json.loads(result)
    # Assert the retrieved data is correct
    expected_result = {'Fire': [2023, 2034, 3004]}
    self.assertEqual(result, expected_result,
                     f'Expected {expected_result}, but got {result}')

  def test_reds_json_get_set(self):
    """Tests the Redis cache."""

    # Configuration for the Redis cache
    config = {'redis_host': 'localhost', 'redis_port': 1234, 'redis_db': 0}
    # Initialize the Redis cache
    self.cache = RedisCache(config=config)

    self.assertIsNotNone(
        self.cache.get_redis_client(),
        'Cache client is not found , check your redis connection !!',
    )

    # Set data in the Redis cache
    self.cache.json_set('testing:test:1', {'Fire': [2023, 2034, 3004]})

    # Retrieve data from the Redis cache
    result = self.cache.json_get('testing:test:1')

    # Assert the retrieved data is correct
    expected_result = {'Fire': [2023, 2034, 3004]}
    self.assertEqual(result, expected_result,
                     f'Expected {expected_result}, but got {result}')

  def test_redis_via_factory(self):

    self.cache = CacheFactory.get_cache('redis', {
        'redis_host': 'localhost',
        'redis_port': 1234,
        'redis_db': 0
    })
    self.cache.json_set('testing:factory:0', json.dumps({'welcome': 'back'}))
    self.assertEqual({'welcome': 'back'},
                     json.loads(self.cache.json_get('testing:factory:0')))

  def test_redis_bin_get_set(self):
    # Configuration for the Redis cache
    config = {'redis_host': 'localhost', 'redis_port': 1234, 'redis_db': 0}
    # Initialize the Redis cache
    self.cache = RedisCache(config=config)

    self.assertIsNotNone(
        self.cache.get_redis_client(),
        'Cache client is not found , check your redis connection !!',
    )

    # Set data in the Redis cache
    self.cache.bin_set(
        'testing:test:2',
        {'Fire': [2023, 2034, 3004]},
    )

    # Retrieve data from the Redis cache
    result = self.cache.bin_get('testing:test:2')

    # Assert the retrieved data is correct
    expected_result = {'Fire': [2023, 2034, 3004]}
    self.assertEqual(result, expected_result,
                     f'Expected {expected_result}, but got {result}')


if __name__ == '__main__':
  unittest.main(testRunner=ColorfulTestRunner())
