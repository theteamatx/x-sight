import subprocess
import time
import unittest

from helpers.logs.logs_handler import logger as logging
import redis
from tests.colorful_tests import ColorfulTestRunner


class RedisContainerTest(unittest.TestCase):
  redis_client = None

  @classmethod
  def wait_for_redis(cls, host, port, timeout=30):
    start_time = time.time()
    while time.time() - start_time < timeout:
      try:
        cls.redis_client = redis.StrictRedis(host=host, port=port)
        cls.redis_client.ping()
        logging.info('Redis is ready!')
        return
      except redis.ConnectionError:
        time.sleep(1)
    raise TimeoutError(f'Redis not ready after {timeout} seconds')

  @classmethod
  def _end_container(cls):
    """Stops the Docker containers."""
    if cls.redis_client:
      keys_to_delete = cls.redis_client.keys('testing:*')
      if keys_to_delete:
        cls.redis_client.delete(*keys_to_delete)  # Delete all matching keys
    try:
      logging.info('Stopping Docker containers ...')
      subprocess.run(['docker-compose', 'down'], check=True)
      logging.info('Docker containers stopped successfully...')
    except subprocess.CalledProcessError as e:
      logging.info(f'Failed to stop Docker containers : {e}')
      raise e

  @classmethod
  def setUpClass(cls):
    cls.cache = None
    cls._end_container()
    try:
      logging.info('Starting Docker containers ...')
      subprocess.run(['docker-compose', 'up', '-d'], check=True)
      # Wait for Redis to be ready
      cls.wait_for_redis('localhost', 1234)
      logging.info('Docker containers started successfully...')
    except subprocess.CalledProcessError as e:
      logging.info(f'Failed to start Docker containers : {e}')
      raise e

  @classmethod
  def tearDownClass(cls):
    cls._end_container()


if __name__ == '__main__':
  unittest.main(testRunner=ColorfulTestRunner())
