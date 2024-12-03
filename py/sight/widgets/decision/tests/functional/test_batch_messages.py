"""Tests for CachedBatchMessages."""

import threading
from typing import Dict
import unittest

from sight.widgets.decision.shared_batch_messages import CachedBatchMessages
from sight.widgets.decision.shared_batch_messages import DecisionMessage
from sight_service.tests import colorful_tests


class TestCachedBatchMessages(unittest.TestCase):

  def setUp(self):
    self.instance1 = CachedBatchMessages.get_instance(sight_id="sight1",
                                                      thread_id="thread1")
    self.instance2 = CachedBatchMessages.get_instance(sight_id="sight2",
                                                      thread_id="thread2")
    self.decision_message = DecisionMessage(
        action_params={"param1": "value1"},
        action_id=1,
        reward=10,
        discount=5,
        outcome_params={"outcome1": "result1"},
    )

  def tearDown(self):
    self.instance1.clear()
    self.instance2.clear()

  def test_singleton_behavior(self):
    instance3 = CachedBatchMessages.get_instance(sight_id="sight1",
                                                 thread_id="thread1")
    self.assertIs(self.instance1, instance3)

  def test_set_and_get(self):
    self.instance1.set(1, self.decision_message)
    retrieved_message = self.instance1.get(1)
    self.assertEqual(retrieved_message, self.decision_message)

  def test_update_message(self):
    self.instance1.set(1, self.decision_message)
    self.instance1.update(1, reward=20, discount=10)
    updated_message = self.instance1.get(1)
    self.assertEqual(updated_message.reward, 20)
    self.assertEqual(updated_message.discount, 10)

  def test_delete_message(self):
    self.instance1.set(1, self.decision_message)
    self.instance1.delete(1)
    self.assertIsNone(self.instance1.get(1))

  def test_clear_cache(self):
    self.instance1.set(1, self.decision_message)
    self.instance1.set(2, self.decision_message)
    self.instance1.clear()
    self.assertEqual(self.instance1.all_messages(), {})

  def test_thread_safety(self):

    def add_messages(instance, start, end):
      for i in range(start, end):
        instance.set(i, self.decision_message)

    threads = [
        threading.Thread(target=add_messages, args=(self.instance1, 0, 50)),
        threading.Thread(target=add_messages, args=(self.instance1, 50, 100)),
    ]

    for thread in threads:
      thread.start()

    for thread in threads:
      thread.join()

    self.assertEqual(len(self.instance1.all_messages()), 100)


if __name__ == "__main__":
  unittest.main(testRunner=colorful_tests.ColorfulTestRunner())
