import datetime
import json
import os
import shutil
import unittest

from sight_service.message_logger import LogStorageCollectStrategy
from sight_service.tests import colorful_tests

datetime = datetime.datetime


class TestCacheBasedLogStorageAndCollectorLocal(unittest.TestCase):

  def setUp(self):
    # Setup for local cache
    config = {
        "local_base_dir": f"/tmp/test_logs",
        "dir_prefix": "test_log_chunks/",
    }
    # Create instances of the LogStorageStrategy and LogCollector
    self.log_storage_collect_strategy = LogStorageCollectStrategy(
        cache_type='local', config=config)

    self.local_base_dir = os.path.abspath(config["local_base_dir"])

  def tearDown(self):
    # Cleanup after tests
    if os.path.exists(self.local_base_dir):
      shutil.rmtree(self.local_base_dir)

  def test_save_and_collect_single_log(self):
    """Test saving a single log and collecting it."""
    log_data = [{"state": "processed", "message_id": "123"}]

    # Save log
    self.log_storage_collect_strategy.save_logs(log_data)

    # Collect logs
    collected_logs = self.log_storage_collect_strategy.collect_logs()

    # Assert
    self.assertEqual(collected_logs, log_data,
                     "Collected logs should match the saved logs.")

  def test_save_and_collect_multiple_logs(self):
    """Test saving multiple logs and collecting them."""
    log_data1 = [{"state": "processed", "message_id": "123"}]
    log_data2 = [{"state": "failed", "message_id": "456"}]

    # Save logs
    self.log_storage_collect_strategy.save_logs(log_data1)
    self.log_storage_collect_strategy.save_logs(log_data2)

    # Collect logs
    collected_logs = self.log_storage_collect_strategy.collect_logs()

    # Assert
    self.assertEqual(
        collected_logs,
        log_data1 + log_data2,
        "Collected logs should match the concatenated saved logs.",
    )

  def test_collect_logs_handles_empty_directory(self):
    """Test collecting logs when no logs have been saved."""
    # Collect logs without saving any
    collected_logs = self.log_storage_collect_strategy.collect_logs()

    # Assert
    self.assertEqual(
        collected_logs,
        [],
        "Collected logs should be empty if no logs are saved.",
    )

  def test_collect_logs_handles_partial_logs(self):
    """Test collecting logs when some files exist."""
    log_data = [{"state": "queued", "message_id": "789"}]

    # Save logs
    self.log_storage_collect_strategy.save_logs(log_data)

    # Manually create an additional file that isn't valid JSON
    self.log_storage_collect_strategy.cache.json_set('logs_chunks:invalid.json',
                                                     ['INVALID JSON'])

    # Collect logs
    collected_logs = self.log_storage_collect_strategy.collect_logs()

    # Assert
    self.assertEqual(
        collected_logs,
        log_data,
        "Collector should ignore invalid files and return only valid logs.",
    )

  def test_save_and_collect_large_number_of_logs(self):
    """Test saving and collecting a large number of logs."""
    num_logs = 1000
    logs = [{
        "state": "processed",
        "message_id": f"{i}"
    } for i in range(num_logs)]

    # Save logs in chunks
    for log in logs:
      self.log_storage_collect_strategy.save_logs([log])

    # Collect logs
    collected_logs = self.log_storage_collect_strategy.collect_logs()

    # Assert
    self.assertEqual(
        len(collected_logs),
        num_logs,
        "Collected logs should match the number of saved logs.",
    )
    self.assertEqual(sorted(collected_logs, key=lambda x: json.dumps(x)),
                     sorted(logs, key=lambda x: json.dumps(x)))


if __name__ == '__main__':
  unittest.main(testRunner=colorful_tests.ColorfulTestRunner())
