"""A message queue implementation using reader-writer locks."""

import abc
import copy
from typing import Callable, Dict, Generic, Optional, Protocol, TypeVar
import uuid

from overrides import overrides
from readerwriterlock import rwlock


class UUIDStrategy(abc.ABC):

  @abc.abstractmethod
  def generate_id(self) -> int:
    pass


class IncrementalUUID(UUIDStrategy):

  def __init__(self):
    self.current_id = 1

  def generate_id(self) -> int:
    unique_id = self.current_id
    self.current_id += 1
    return unique_id


class RandomUUID(UUIDStrategy):

  def generate_id(self) -> int:
    return uuid.uuid4().int  # Using the integer representation of UUID


# Define a generic type variable for messages
T = TypeVar('T')


class IMessageQueue(Protocol, Generic[T]):
  """A message queue is a data structure that stores messages."""

  def push_message(self, message: T) -> int:
    """Pushes a message to the queue.

    Args:
      message: The message to push.
    """
    ...

  def process_messages(
      self, worker_id: str, new_batch_size: Optional[int] = None
      ) -> Dict[int, T]:
    """Processes a batch of messages for a given worker.

    Args:
      worker_id: The ID of the worker that will process the messages.
      new_batch_size: The size of the batch to process. If not provided, the
        default batch size will be used.
    """
    ...

  def complete_message(
      self, message_id: str, worker_id: str, extra_details: Optional[any] = None
    ) -> None:
    """Completes a message of the given message ID of the given worker it moves it to the completed queue.

    Args:
      message_id: The ID of the message to complete.
      worker_id: The ID of the worker that completed the message.
      extra_details: Any extra details to add to the completed message. (only
        for dict types messages)
    """
    ...

  def get_status(self) -> Dict[str, int]:
    """Returns the status of the message queue."""
    ...

  def get_all_messages(self) -> Dict[str, Dict[int, T]]:
    """Returns all messages in the message queue."""
    ...

  def find_message_location(self, message_id: str) -> str:
    """Returns the location of the message in the message queue."""
    ...


class MessageQueue(IMessageQueue[T]):
  """A message queue is a data structure that stores messages.

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

  def __init__(
      self,
      id_generator: UUIDStrategy,
      batch_size: int = 1,
      lock_factory: Callable[[], rwlock.RWLockFairD] = rwlock.RWLockFairD,
  ):
    self.id_generator = id_generator
    self.pending: Dict[int, T] = {}
    self.active: Dict[str, Dict[int, T]] = {}
    self.completed: Dict[int, T] = {}

    self.batch_size = batch_size
    self.pending_lock = lock_factory()
    self.active_lock = lock_factory()
    self.completed_lock = lock_factory()

  def __str__(self) -> str:
    all_messages = self.get_all_messages()

    result = ['MessageQueue:']
    result.append('  Pending Messages:')
    for id, message in all_messages['pending'].items():
      result.append(f'    ID: {id}, Message: {message}')

    result.append('  Active Messages:')
    for id, message in all_messages['active'].items():
      result.append(f'    ID: {id}, Message: {message}')

    result.append('  Completed Messages:')
    for id, message in all_messages['completed'].items():
      result.append(f'    ID: {id}, Message: {message}')

    return '\n'.join(result)

  # @overrides
  def push_message(self, message: T) -> str:
    """Pushes a message to the queue.

    Args:
      message: The message to push.
    """
    unique_id = self.id_generator.generate_id()
    with self.pending_lock.gen_wlock():
      self.pending[unique_id] = message
    return unique_id

  # @overrides
  def process_messages(
      self, worker_id: str, new_batch_size: Optional[int] = None
    ) -> Dict[int, T]:
    """Processes a batch of messages for a given worker.

    Args:
      worker_id: The ID of the worker that will process the messages.
      new_batch_size: The size of the batch to process. If not provided, the
        default batch size will be used.
    """
    batch_size = (
        new_batch_size if new_batch_size is not None else self.batch_size
    )
    batch: Dict[int, T] = {}

    with self.pending_lock.gen_wlock():
      for _ in range(min(batch_size, len(self.pending))):
        message_id = next(iter(self.pending))
        message = self.pending.pop(message_id)
        batch[message_id] = message

    with self.active_lock.gen_wlock():
      if worker_id not in self.active:
        self.active[worker_id] = {}
      self.active[worker_id].update(batch)

    return batch

  # @overrides
  def complete_message(
      self, message_id: str, worker_id: str, extra_details: Optional[any] = None
    ) -> None:
    """Completes a message of the given message ID of the given worker it moves it to the completed queue.

    Args:
      message_id: The ID of the message to complete.
      worker_id: The ID of the worker that completed the message.
      extra_details: Any extra details to add to the completed message. (only
        for dict types messages)
    """
    with self.active_lock.gen_wlock():
      if message_id in self.active.get(worker_id, {}):
        message = copy.deepcopy(self.active[worker_id][message_id])
        del self.active[worker_id][message_id]

        with self.completed_lock.gen_wlock():
          self.completed[message_id] = message
          if extra_details is not None and isinstance(
              self.completed[message_id], dict
          ):
            self.completed[message_id].update(extra_details)
      else:
        raise ValueError(
            f'Message ID {message_id} not found for worker {worker_id}'
        )

  # @overrides
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

  # @overrides
  def get_all_messages(self) -> Dict[str, Dict[int, T]]:
    """Returns all messages in the message queue."""
    with self.pending_lock.gen_rlock():
      pending_copy = copy.deepcopy(self.pending)
    with self.active_lock.gen_rlock():
      active_copy = copy.deepcopy(self.active)
    with self.completed_lock.gen_rlock():
      completed_copy = copy.deepcopy(self.completed)

    return {
        'pending': pending_copy,
        'active': active_copy,
        'completed': completed_copy,
    }

  # @overrides
  def find_message_location(self, message_id: str) -> str:
    """Returns the location of the message in the message queue."""
    with self.pending_lock.gen_rlock():
      if message_id in self.pending:
        return 'pending'

    with self.active_lock.gen_rlock():
      for _, messages in self.active.items():
        if message_id in messages:
          return 'active'

    with self.completed_lock.gen_rlock():
      if message_id in self.completed:
        return 'completed'

    return 'not found'
