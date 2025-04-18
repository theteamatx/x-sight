"""Tests for the Redis cache."""

import subprocess
import time
import unittest

from helpers.cache.cache_factory import CacheFactory
from helpers.cache.cache_redis import RedisCache
from helpers.logs.logs_handler import logger as logging
import redis
from tests.colorful_tests import ColorfulTestRunner


class CacheRedisTest(unittest.TestCase):
  """Tests for the Redis cache."""

  def wait_for_redis(self, host, port, timeout=30):
    start_time = time.time()
    while time.time() - start_time < timeout:
      try:
        client = redis.StrictRedis(host=host, port=port)
        client.ping()
        logging.info('Redis is ready!')
        return
      except redis.ConnectionError:
        time.sleep(1)
    raise TimeoutError(f'Redis not ready after {timeout} seconds')

  def _end_container(self):
    """Stops the Docker containers."""
    if self.cache and self.cache.get_redis_client():
      client = self.cache.get_redis_client()
      keys_to_delete = client.keys('testing:*')
      if keys_to_delete:
        client.delete(*keys_to_delete)  # Delete all matching keys
    try:
      logging.info('Stopping Docker containers ...')
      subprocess.run(['docker-compose', 'down'], check=True)
      logging.info('Docker containers stopped successfully...')
    except subprocess.CalledProcessError as e:
      logging.info(f'Failed to stop Docker containers : {e}')
      raise e

  def tearDown(self):
    super().tearDown()
    self._end_container()

  def setUp(self):
    super().setUp()
    self.cache = None
    self._end_container()
    try:
      logging.info('Starting Docker containers ...')
      subprocess.run(['docker-compose', 'up', '-d'], check=True)
      # Wait for Redis to be ready
      self.wait_for_redis('localhost', 1234)
      logging.info('Docker containers started successfully...')
    except subprocess.CalledProcessError as e:
      logging.info(f'Failed to start Docker containers : {e}')
      raise e

  def test_redis_cache(self):
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
    self.cache.json_set(
        'testing:ACR203:2013:FVS:MANAGED:FIRE_0001011100',
        {'Fire': [2023, 2034, 3004]},
    )

    # Retrieve data from the Redis cache
    result = self.cache.json_get(
        'testing:ACR203:2013:FVS:MANAGED:FIRE_0001011100')

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
    self.cache.json_set('testing:KEY', {'welcome': 'back'})
    self.assertEqual({'welcome': 'back'}, self.cache.json_get('testing:KEY'))

  def test_bin_data(self):
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
        'testing:ACR203:2013:FVS:MANAGED:FIRE_0001011100',
        {'Fire': [2023, 2034, 3004]},
    )

    # Retrieve data from the Redis cache
    result = self.cache.bin_get(
        'testing:ACR203:2013:FVS:MANAGED:FIRE_0001011100')

    # Assert the retrieved data is correct
    expected_result = {'Fire': [2023, 2034, 3004]}
    self.assertEqual(result, expected_result,
                     f'Expected {expected_result}, but got {result}')


if __name__ == '__main__':
  unittest.main(testRunner=ColorfulTestRunner())
