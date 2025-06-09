"""Tests for the LogStorageCollectStrategy."""

import datetime
import json
import unittest

from google.cloud import storage
from sight_service.message_queue.message_logger.log_storage_collect import (
    CachedBasedLogStorageCollectStrategy
)
from sight_service.tests import colorful_tests

datetime = datetime.datetime


class TestCacheBasedLogStorageAndCollectorGCS(unittest.TestCase):
  """Tests for the LogStorageCollectStrategy using a GCS cache."""

  def _delete_gcs_folder(self, bucket_name: str, folder_name: str):
    """Deletes all objects in a folder in a Google Cloud Storage bucket."""
    client = storage.Client()
    bucket = client.bucket(bucket_name)

    # Add a trailing slash if missing to ensure only matching "folder" objects
    # are deleted
    if not folder_name.endswith("/"):
      folder_name += "/"

    # List all objects with the specified prefix
    blobs = bucket.list_blobs(prefix=folder_name)

    # Delete each object
    for blob in blobs:
      print(f"Deleting: {blob.name}")
      blob.delete()

    print(f"All objects in folder '{folder_name}' have been deleted from bucket"
          f" '{bucket_name}'.")

  def setUp(self):
    super().setUp()
    # Setup for local cache
    config = {
        "gcs_base_dir": "test_sight_cache_0/testing_logs",
        "gcs_bucket": "cameltrain-sight",
        "dir_prefix": "test_log_chunks/",
    }
    self.config = config
    self.log_storage_collect_strategy = CachedBasedLogStorageCollectStrategy(
        cache_type="gcs", config=config)

  def tearDown(self):
    super().tearDown()
    self._delete_gcs_folder(
        bucket_name=self.config["gcs_bucket"],
        folder_name=self.config["gcs_base_dir"],
    )

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
    self.log_storage_collect_strategy.cache.set("logs_chunks:invalid.json",
                                                json.dumps(["INVALID JSON"]))

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
    num_logs = 50
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
    self.assertEqual(
        sorted(collected_logs, key=json.dumps),
        sorted(logs, key=json.dumps),
    )

    # Assert
    self.assertEqual(
        len(collected_logs),
        num_logs,
        "Collected logs should match the number of saved logs.",
    )
    self.assertEqual(
        sorted(collected_logs, key=json.dumps),
        sorted(logs, key=json.dumps),
    )


if __name__ == "__main__":
  unittest.main(testRunner=colorful_tests.ColorfulTestRunner())
