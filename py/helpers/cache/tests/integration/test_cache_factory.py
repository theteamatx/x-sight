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

  def test_redis_local(self):

    self.cache = CacheFactory.get_cache('redis_local')
    excepted_key = 'key'
    excepted_value = 'value'
    self.cache.set(excepted_key, excepted_value)
    value = self.cache.get(excepted_key)
    self.assertEqual(value, excepted_value, 'Our values got mis-matched')


if __name__ == '__main__':
  unittest.main(testRunner=ColorfulTestRunner())
