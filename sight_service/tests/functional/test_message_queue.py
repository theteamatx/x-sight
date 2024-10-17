"""Tests for the MessageQueue class."""

import unittest

import sight_service.message_queue as mq
from sight_service.tests import colorful_tests


class TestMessageQueue(unittest.TestCase):
  """Tests for MessageQueue class.

  Attributes:
    incremental_id_generator: IncrementalUUID()
    queue: MessageQueue[int]
  """

  def setUp(self):
    """Set up the MessageQueue and IncrementalUUID for testing."""
    super().setUp()
    # Use IncrementalUUID for most tests to have predictable IDs
    self.incremental_id_generator = mq.IncrementalUUID()
    self.queue = mq.MessageQueue[int](
        id_generator=self.incremental_id_generator, batch_size=2
    )

  def test_add_message_with_incremental_id(self):
    message_id = self.queue.push_message(100)
    self.assertEqual(message_id, 1)
    status = self.queue.get_status()
    self.assertEqual(status['pending'], 1)
    self.assertEqual(status['active'], 0)
    self.assertEqual(status['completed'], 0)

  def test_process_messages(self):
    """Test process_messages() with a batch size of 2 and 3 messages in the queue."""
    self.queue.push_message(100)
    self.queue.push_message(200)
    self.queue.push_message(300)

    batch = self.queue.create_active_batch(worker_id='worker1')
    self.assertEqual(len(batch), 2)
    self.assertIn(1, batch)
    self.assertIn(2, batch)

    status = self.queue.get_status()
    self.assertEqual(status['pending'], 1)
    self.assertEqual(status['active'], 2)
    self.assertEqual(status['completed'], 0)

  def test_complete_message(self):
    """Test complete_message() with a batch size of 2 and 2 messages in the queue."""
    self.queue.push_message(100)
    self.queue.push_message(200)

    batch = self.queue.create_active_batch(worker_id='worker1')
    self.assertEqual(len(batch), 2)
    self.queue.complete_message(1, 'worker1')

    status = self.queue.get_status()
    self.assertEqual(status['pending'], 0)
    self.assertEqual(status['active'], 1)
    self.assertEqual(status['completed'], 1)
    self.assertIn(1, self.queue.get_all_messages()['completed'])

  def test_complete_message_with_lambda_update(self):
    self.queue.push_message(500)
    batch = self.queue.create_active_batch(worker_id='worker2')
    self.assertEqual(len(batch), 1)

    # Apply a lambda function that doubles the message value
    update_fn = lambda msg: msg * 2
    self.queue.complete_message(1, 'worker2', update_fn)

    # Verify that the message was updated using the lambda function
    completed_msg = self.queue.get_completed()[1]
    self.assertEqual(completed_msg, 1000)  # 500 * 2

  def test_get_pending_messages(self):
    self.queue.push_message(100)
    self.queue.push_message(200)
    pending_messages = self.queue.get_pending()
    self.assertEqual(len(pending_messages), 2)

  def test_get_active_messages(self):
    self.queue.push_message(100)
    self.queue.push_message(200)
    self.queue.create_active_batch(worker_id='worker1')
    active_messages = self.queue.get_active()
    self.assertIn('worker1', active_messages)
    self.assertEqual(len(active_messages['worker1']), 2)

  def test_get_completed_messages(self):
    self.queue.push_message(100)
    self.queue.create_active_batch(worker_id='worker1')
    self.queue.complete_message(1, 'worker1')
    completed_messages = self.queue.get_completed()
    self.assertEqual(len(completed_messages), 1)

  def test_process_and_complete_message(self):
    """Test process_messages() and complete_message() with a batch size of 1."""
    self.queue.push_message(100)
    self.queue.push_message(200)
    self.queue.push_message(300)

    batch = self.queue.create_active_batch(
        worker_id='worker1', new_batch_size=1
    )
    self.assertEqual(len(batch), 1)
    self.assertIn(1, batch)

    self.queue.complete_message(1, 'worker1')
    status = self.queue.get_status()
    self.assertEqual(status['pending'], 2)
    self.assertEqual(status['active'], 0)
    self.assertEqual(status['completed'], 1)

  def test_complete_message_not_found(self):
    self.queue.push_message(100)
    self.queue.push_message(200)

    batch = self.queue.create_active_batch(worker_id='worker1')
    self.assertEqual(len(batch), 2)
    with self.assertRaises(ValueError):
      self.queue.complete_message(999, 'worker1')

  def test_empty_process_messages(self):
    batch = self.queue.create_active_batch(worker_id='worker1')
    self.assertEqual(len(batch), 0)
    status = self.queue.get_status()
    self.assertEqual(status['pending'], 0)
    self.assertEqual(status['active'], 0)
    self.assertEqual(status['completed'], 0)

  def test_find_message_location(self):
    self.queue.push_message(100)
    self.queue.push_message(200)
    self.queue.create_active_batch(worker_id='worker1')
    location = self.queue.find_message_location(1)
    self.assertEqual(location, mq.MessageLocation.ACTIVE)

  def test_get_all_messages(self):
    """Test get_all_messages() with 2 pending messages and 1 completed message."""

    self.queue.push_message(100)
    self.queue.push_message(200)

    all_messages = self.queue.get_all_messages()
    self.assertEqual(len(all_messages['pending']), 2)
    self.assertEqual(
        len(all_messages['active']), 0
    )  # No messages should be in active yet
    self.assertEqual(len(all_messages['completed']), 0)

    # Process the messages, which should move them
    # to 'active' under a specific worker_id
    self.queue.create_active_batch(worker_id='worker1')
    all_messages = self.queue.get_all_messages()

    # After processing, 'pending' should be empty, 'active'
    # should have 2 messages under 'worker1'
    self.assertEqual(len(all_messages['pending']), 0)
    self.assertIn('worker1', all_messages['active'])
    self.assertEqual(len(all_messages['active']['worker1']), 2)
    self.assertEqual(len(all_messages['completed']), 0)

    self.queue.complete_message(1, 'worker1')
    all_messages = self.queue.get_all_messages()

    # 'pending' should still be empty, 'active' should have
    # 1 message under 'worker1', and 'completed' should have 1 message
    self.assertEqual(len(all_messages['pending']), 0)
    self.assertIn('worker1', all_messages['active'])
    self.assertEqual(len(all_messages['active']['worker1']), 1)
    self.assertEqual(len(all_messages['completed']), 1)

  def test_add_message_with_uuid(self):
    """Test add_message() with a UUID ID generator."""
    uuid_id_generator = mq.RandomUUID()
    queue_with_uuid = mq.MessageQueue[str](
        id_generator=uuid_id_generator, batch_size=2
    )

    message_id1 = queue_with_uuid.push_message('Task A')
    message_id2 = queue_with_uuid.push_message('Task B')

    # Check that the UUIDs are unique and have been assigned correctly
    self.assertNotEqual(message_id1, message_id2)
    self.assertIn(message_id1, queue_with_uuid.get_all_messages()['pending'])
    self.assertIn(message_id2, queue_with_uuid.get_all_messages()['pending'])


if __name__ == '__main__':
  unittest.main(testRunner=colorful_tests.ColorfulTestRunner())
