"""Tests for CachedBatchMessages."""

import json
import unittest

from sight_service import shared_batch_messages
from sight_service.tests import colorful_tests


class TestCachedBatchMessages(unittest.TestCase):
  """Tests for the CachedBatchMessages class.

  Attributes:
    cache:  The CachedBatchMessages object to test.
  """

  def setUp(self):
    """Sets up the CachedBatchMessages object for testing."""
    super().setUp()
    self.cache = shared_batch_messages.CachedBatchMessages()

  def test_set_and_get_message(self):
    msg = shared_batch_messages.DecisionMessage(action_params={"key": "value"},
                                                action_id=1)
    self.cache.set(1, json.dumps(msg))

    retrieved_msg = json.loads(self.cache.get(1))
    self.assertIsNotNone(retrieved_msg)
    self.assertEqual(retrieved_msg.action_id, 1)

  def test_get_nonexistent_message(self):
    self.assertIsNone(self.cache.get(999))

  def test_update_message(self):
    msg = shared_batch_messages.DecisionMessage(action_params={"key": "value"},
                                                action_id=1)
    self.cache.set(1, json.dumps(msg))

    self.cache.update(1, reward=20, discount=5)
    updated_msg = json.loads(self.cache.get(1))

    self.assertEqual(updated_msg.reward, 20)
    self.assertEqual(updated_msg.discount, 5)

  def test_update_nonexistent_message(self):
    with self.assertRaises(KeyError):
      self.cache.update(999, reward=20)

  def test_update_invalid_field(self):
    msg = shared_batch_messages.DecisionMessage(action_params={"key": "value"},
                                                action_id=1)
    self.cache.set(1, json.dumps(msg))

    with self.assertRaises(AttributeError):
      self.cache.update(1, invalid_field="value")

  def test_delete_message(self):
    msg = shared_batch_messages.DecisionMessage(action_params={"key": "value"},
                                                action_id=1)
    self.cache.set(1, json.dumps(msg))

    self.cache.delete(1)
    self.assertIsNone(self.cache.get(1))

  def test_delete_nonexistent_message(self):
    # Deleting a non-existent message should not raise an error
    self.cache.delete(999)

  def test_all_messages(self):
    """Tests that all_messages returns all messages in the cache."""
    msg1 = shared_batch_messages.DecisionMessage(
        action_params={"key1": "value1"}, action_id=1)
    msg2 = shared_batch_messages.DecisionMessage(
        action_params={"key2": "value2"}, action_id=2)

    self.cache.set(1, json.dumps(msg1))
    self.cache.set(1, json.dumps(msg2))

    all_messages = self.cache.all_messages()
    self.assertEqual(len(all_messages), 2)
    self.assertEqual(all_messages[1].action_id, 1)
    self.assertEqual(all_messages[2].action_id, 2)

  def test_clear_cache(self):
    """Tests that clear_cache clears the cache."""
    msg1 = shared_batch_messages.DecisionMessage(
        action_params={"key1": "value1"}, action_id=1)
    msg2 = shared_batch_messages.DecisionMessage(
        action_params={"key2": "value2"}, action_id=2)

    self.cache.set(1, json.dumps(msg1))
    self.cache.set(1, json.dumps(msg2))

    self.cache.clear()
    self.assertEqual(len(self.cache.all_messages()), 0)


if __name__ == "__main__":
  unittest.main(testRunner=colorful_tests.ColorfulTestRunner())
