"""A list lock message queue implementation using reader-writer locks."""

import copy
import time
from typing import Any, Callable, Dict, Optional, TypeVar

from helpers.logs.logs_handler import logger as logging
from overrides import overrides
from readerwriterlock import rwlock
from sight_service.message_queue.message_logger.interface import (
    ILogStorageCollectStrategy
)
from sight_service.message_queue.message_logger.log_storage_collect import (
    NoneLogStorageCollectStrategy
)
from sight_service.message_queue.message_logger.message_logger import (
    MessageFlowLogger
)
from sight_service.message_queue.mq_interface import ID
from sight_service.message_queue.mq_interface import IMessageQueue
from sight_service.message_queue.mq_interface import IncrementalUUID
from sight_service.message_queue.mq_interface import IUUIDStrategy
from sight_service.message_queue.mq_interface import MessageState

# Define a generic type variable for messages
T = TypeVar('T')


class ListLockMessageQueue(IMessageQueue[T]):
  """A message queue is a data structure that stores messages.

    ##### State machine for each message:
    ##### 1. NotInQueue -> The message does not exist in the queue.
    ##### 2. Pending -> The message is added to the queue but not yet processed.
    ##### 3. Active -> The message is assigned to a worker for processing.
    ##### 4. Completed -> The message is processed and moved to the completed
    state.


  Attributes:
    id_generator: The ID generator used to generate unique IDs for messages.
    pending: A dictionary of pending messages, keyed by message ID.
    active: A dictionary of active messages, keyed by worker ID and message ID.
    completed: A dictionary of completed messages, keyed by message ID.
    batch_size: The size of the batch to process. (default: 1)
    pending_lock: The lock used to synchronize access to the pending messages.
    active_lock: The lock used to synchronize access to the active messages.
    completed_lock: The lock used to synchronize access to the completed
      messages.
  """

  # Locking Procedure:
  # -----------------
  # This class uses a Reader-Writer locking mechanism to protect access to
  # shared resources (the message queues).
  # - The reader-writer locks allow concurrent reads but ensure exclusive access
  #   for writes, improving efficiency when multiple reads are performed
  #   simultaneously.
  # - There are three distinct locks used:
  #   1. `pending_lock` - Protects the `pending` messages dictionary.
  #   2. `active_lock` - Protects the `active` messages dictionary.
  #   3. `completed_lock` - Protects the `completed` messages dictionary.
  # - Each lock is associated with a specific queue state to ensure that
  #   operations on each state are thread-safe and do not interfere with each
  #   other.

  def __init__(
      self,
      id_generator: IUUIDStrategy = IncrementalUUID(),
      batch_size: int = 1,
      lock_factory: Callable[[], rwlock.RWLockFairD] = rwlock.RWLockFairD,
      logger_storage_strategy:
      ILogStorageCollectStrategy = NoneLogStorageCollectStrategy,
  ):

    if logger_storage_strategy is None:
      logger_storage_strategy = NoneLogStorageCollectStrategy

    self.id_generator = id_generator
    self.pending: Dict[ID, T] = {}
    self.active: Dict[str, Dict[ID, T]] = {}
    self.completed: Dict[ID, T] = {}

    self.batch_size = batch_size
    self.pending_lock = lock_factory()
    self.active_lock = lock_factory()
    self.completed_lock = lock_factory()

    # logger
    self.logger = MessageFlowLogger(storage_strategy=logger_storage_strategy)

  def __str__(self) -> str:
    messages_status = self.get_status()
    result = ['MessageQueue:']
    result.append('  Pending Messages:')
    result.append(f'    Messages 📩 : {messages_status["pending"]}')
    # for msg_id, message in all_messages['pending'].items():
    # result.append(f'    ID: {msg_id}, Message: {message}')

    result.append('  Active Messages:')
    result.append(f'    Messages 📨 : {messages_status["active"]}')

    for worker_id, messages in self.get_active().items():
      result.append(f'    ID: {worker_id}, Messages 📨: {len(messages)}')

    result.append('  Completed Messages:')
    result.append(f'    Messages ✉️ : {messages_status["completed"]}')

    # for msg_id, message in all_messages['completed'].items():
    #   result.append(f'    ID: {msg_id}, Message: {message}')

    return '\n'.join(result)

  @overrides
  def push_message(self, message: T) -> ID:
    """Pushes a message to the queue.

    Args:
      message: The message to push.

    Returns:
      The unique ID of the pushed message.
    """
    start_time = time.time()

    unique_id = self.id_generator.generate_id()
    with self.pending_lock.gen_wlock():
      self.pending[unique_id] = message

    time_taken_in_second = time.time() - start_time
    # log the message to logger
    self.logger.log_message_state(state='pending',
                                  message_id=unique_id,
                                  time_taken=time_taken_in_second)
    return unique_id

  @overrides
  def create_active_batch(self,
                          worker_id: str,
                          new_batch_size: Optional[int] = None) -> Dict[ID, T]:
    """Move a batch of messages for a given worker into active list.

    Args:
      worker_id: The ID of the worker that will process the messages.
      new_batch_size: The size of the batch to process. If not provided, the
        default batch size will be used.

    Returns:
      A dictionary of messages that were processed, keyed by message ID.
    """
    batch_size = (new_batch_size
                  if new_batch_size is not None else self.batch_size)
    batch: Dict[ID, T] = {}

    start_time = time.time()

    with self.pending_lock.gen_wlock():
      for _ in range(min(batch_size, len(self.pending))):
        message_id = next(iter(self.pending))
        message = self.pending.pop(message_id)
        batch[message_id] = message

    with self.active_lock.gen_wlock():
      if worker_id not in self.active:
        self.active[worker_id] = {}
      self.active[worker_id].update(batch)

    time_taken_in_second = time.time() - start_time
    ## log the messages to logger
    for message_id in batch:
      self.logger.log_message_state(state='active',
                                    message_id=message_id,
                                    worker_id=worker_id,
                                    time_taken=time_taken_in_second)

    return batch

  @overrides
  def complete_message(self,
                       message_id: ID,
                       worker_id: str,
                       update_fn: Callable[[T], T] = None) -> None:
    """Completes a message of the given message ID of the given worker it moves it to the completed queue.

    Args:
      message_id: The ID of the message to complete.
      worker_id: The ID of the worker that completed the message.
      update_fn: A function that takes the current message and returns the
        updated message.
    """

    start_time = time.time()

    with self.active_lock.gen_wlock():
      if message_id not in self.active.get(worker_id, {}):
        raise ValueError(
            f'Failed while completing the msg ,as Message ID {message_id} not'
            f' found for worker {worker_id}')

      message = self.active[worker_id][message_id]
      del self.active[worker_id][message_id]

      if update_fn is not None:
        logging.info('Before update_fn msg: %s', message)
        message = update_fn(message)  # Apply the lambda to update the message
        logging.info('After update_fn msg: %s', message)

    with self.completed_lock.gen_wlock():
      self.completed[message_id] = message

    time_taken_in_second = time.time() - start_time
    ## log the message to logger
    self.logger.log_message_state(state='completed',
                                  message_id=message_id,
                                  worker_id=worker_id,
                                  time_taken=time_taken_in_second)

  @overrides
  def get_status(self) -> Dict[str, int]:
    """Returns the status of the message queue."""
    with self.pending_lock.gen_rlock():
      pending_len = len(self.pending)
    with self.active_lock.gen_rlock():
      active_len = sum(len(batch) for batch in self.active.values())
    with self.completed_lock.gen_rlock():
      completed_len = len(self.completed)

    return {
        'pending': pending_len,
        'active': active_len,
        'completed': completed_len,
    }

  @overrides
  def get_pending(self) -> Dict[ID, T]:
    """Returns all pending messages in the queue."""
    with self.pending_lock.gen_rlock():
      return copy.copy(self.pending)

  @overrides
  def get_active(self) -> Dict[str, Dict[ID, T]]:
    """Returns all active messages in the queue."""
    with self.active_lock.gen_rlock():
      return copy.copy(self.active)

  @overrides
  def get_completed(self) -> Dict[ID, T]:
    """Returns all completed messages in the queue."""
    with self.completed_lock.gen_rlock():
      return copy.copy(self.completed)

  @overrides
  def is_message_in_pending(self, message_id: ID) -> bool:
    """Returns the true if the message in the pending queue."""
    with self.pending_lock.gen_rlock():
      return message_id in self.pending

  @overrides
  def is_message_in_active(self, message_id: ID) -> bool:
    """Returns the true if the message in the active queue."""
    with self.active_lock.gen_rlock():
      for _, messages in self.active.items():
        return message_id in messages

  @overrides
  def is_message_in_completed(self, message_id: ID) -> bool:
    """Returns the true if the message in the completed queue."""
    with self.completed_lock.gen_rlock():
      return message_id in self.completed

  @overrides
  def find_message_location(self, message_id: ID) -> MessageState:
    """Returns the location of the message in the message queue."""
    with self.pending_lock.gen_rlock():
      if message_id in self.pending:
        return MessageState.PENDING

    with self.active_lock.gen_rlock():
      for _, messages in self.active.items():
        if message_id in messages:
          return MessageState.ACTIVE

    with self.completed_lock.gen_rlock():
      if message_id in self.completed:
        return MessageState.COMPLETED

    return MessageState.NOT_FOUND
