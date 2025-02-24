"""Tests for the MessageFlowLogger."""

import datetime
import unittest

from google.cloud import storage
from sight_service.message_queue.message_logger.log_storage_collect import (
    CachedBasedLogStorageCollectStrategy
)
from sight_service.message_queue.message_logger.message_logger import (
    MessageFlowLogger
)
from sight_service.tests import colorful_tests

datetime = datetime.datetime


class TestMessageFlowLoggerGCS(unittest.TestCase):
  """Tests for the MessageFlowLogger when using a GCS storage strategy."""

  def _delete_gcs_folder(self, bucket_name: str, folder_name: str):
    """Deletes all objects in a GCS folder.

    Args:
      bucket_name: The name of the GCS bucket.
      folder_name: The name of the GCS folder to delete.
    """
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
    # Create instances of the CachedBasedLogStorageCollectStrategy
    # and MessageFlowLogger
    self.log_storage_collect_strategy = CachedBasedLogStorageCollectStrategy(
        cache_type="gcs", config=config)
    self.logger = MessageFlowLogger(
        storage_strategy=self.log_storage_collect_strategy,
        chunk_size=5,
        flush_interval=1,
    )

  def tearDown(self):
    super().tearDown()
    self._delete_gcs_folder(
        bucket_name=self.config["gcs_bucket"],
        folder_name=self.config["gcs_base_dir"],
    )

  def test_message_logging_and_collection(self):
    """Tests that messages are logged and collected correctly."""
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
