"""This module defines the abstract base classes and interfaces for Message Queue and Logger and their strategies."""

import abc
import enum
from typing import Callable, Dict, Generic, Optional, Protocol, TypeVar

# Alias for message ID type
ID = int
# Define a generic type variable for messages
T = TypeVar('T')

abstractmethod = abc.abstractmethod
ABC = abc.ABC


class MessageState(enum.Enum):
  """The state of a message in the message queue."""

  PENDING = 'pending'
  ACTIVE = 'active'
  COMPLETED = 'completed'
  NOT_FOUND = 'not found'

  def __str__(self):
    return str(self.value)


class IUUIDStrategy(ABC):
  """An abstract base class for generating unique IDs.

  This defines a strategy interface for generating unique IDs. Subclasses
  should implement the `generate_id` method to provide different ways of
  creating unique identifiers.
  """

  @abstractmethod
  def generate_id(self) -> ID:
    pass


class IMessageQueue(Protocol, Generic[T]):
  """A message queue is a data structure that stores messages.

  #### State machine for each message:
  ##### 1. NotInQueue -> The message does not exist in the queue.
  ##### 2. Pending -> The message is added to the queue but not yet processed.
  ##### 3. Active -> The message is assigned to a worker for processing.
  ##### 4. Completed -> The message is processed and moved to the completed
  state.
  """

  def push_message(self, message: T) -> ID:
    """Pushes a message to the queue.

    Args:
      message: The message to push.
    """
    ...

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
    ...

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
    ...

  def get_status(self) -> Dict[str, int]:
    """Returns the status of the message queue."""
    ...

  def get_all_messages(self) -> Dict[str, Dict[ID, T]]:
    """Returns all messages in the message queue."""
    ...

  def get_pending(self) -> Dict[ID, T]:
    """Returns all pending messages in the queue."""
    ...

  def get_active(self) -> Dict[str, Dict[ID, T]]:
    """Returns all active messages in the queue."""
    ...

  def get_completed(self) -> Dict[ID, T]:
    """Returns all completed messages in the queue."""
    ...

  def find_message_location(self, message_id: ID) -> MessageState:
    """Returns the location of the message in the message queue."""
    ...

  def is_message_in_pending(self, message_id: ID) -> bool:
    """Checks if the message is in the pending state."""
    ...

  def is_message_in_active(self, message_id: ID) -> bool:
    """Checks if the message is in the active state."""
    ...

  def is_message_in_completed(self, message_id: ID) -> bool:
    """Checks if the message is in the completed state."""
    ...
