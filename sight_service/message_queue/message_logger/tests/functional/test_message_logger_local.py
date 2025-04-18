"""Tests for the LogStorageCollectStrategy and MessageFlowLogger."""

import datetime
import os
import shutil
import unittest

from sight_service.message_queue.message_logger.log_storage_collect import (
    CachedBasedLogStorageCollectStrategy
)
from sight_service.message_queue.message_logger.message_logger import (
    MessageFlowLogger
)
from sight_service.tests import colorful_tests

datetime = datetime.datetime


class TestMessageFlowLoggerLocal(unittest.TestCase):
  """Tests for the LogStorageCollectStrategy and MessageFlowLogger with a local cache."""

  def setUp(self):
    super().setUp()
    # Setup for local cache
    config = {
        "local_base_dir": "/tmp/test_logs",
        "dir_prefix": "test_log_chunks/",
    }
    # Create instances of the CachedBasedLogStorageCollectStrategy
    self.log_storage_collect_strategy = CachedBasedLogStorageCollectStrategy(
        cache_type="local", config=config)
    self.logger = MessageFlowLogger(
        storage_strategy=self.log_storage_collect_strategy,
        chunk_size=5,
        flush_interval=1,
    )
    self.local_base_dir = os.path.abspath(config["local_base_dir"])

  def tearDown(self):
    super().tearDown()
    self.logger.stop()
    # Cleanup after tests
    if os.path.exists(self.local_base_dir):
      shutil.rmtree(self.local_base_dir)

  def test_message_logging_and_collection(self):
    """Tests message logging and collection with a local cache."""
    # Log messages
    for i in range(15):  # 15 messages
      self.logger.log_message_state(
          state="processed",
          message_id=str(i),
          worker_id=f"worker_{i}",
          message_details={"info": f"message_{i}"},
      )

    # Wait for logs to flush
    self.logger.stop()

    # Collect logs
    collected_logs = self.log_storage_collect_strategy.collect_logs()

    # Verify logs
    self.assertEqual(len(collected_logs), 15)
    for i, log in enumerate(collected_logs):
      self.assertEqual(log["message_id"], str(i))
      self.assertEqual(log["state"], "processed")
      self.assertEqual(log["worker_id"], f"worker_{i}")
      self.assertEqual(log["message_details"]["info"], f"message_{i}")


if __name__ == "__main__":
  unittest.main(testRunner=colorful_tests.ColorfulTestRunner())
