"""A module for shared batch messages."""

import dataclasses
import threading
from typing import Any, ClassVar, Dict, Optional

Lock = threading.Lock
dataclass = dataclasses.dataclass


@dataclass
class DecisionMessage:
  """A message that is sent to the agent to make a decision.

  Attributes:
      action_params: A dictionary of parameters for the action.
      action_id: The ID of the action.
      reward: The reward for the action.
      discount: The discount for the action.
      outcome_params: A dictionary of parameters for the outcome.
  """

  action_params: Dict[str, Any]
  action_id: int
  reward: Optional[int] = None
  discount: Optional[int] = None
  outcome_params: Optional[Dict[str, Any]] = None


class CachedBatchMessages:
  """A singleton class for caching batch messages.

  Supports multiple singleton instances distinguished by sight_id and thread_id.

  Attributes:
      batch_messages: A dictionary of cached DecisionMessages.
      _instances: A class-level dictionary to store singleton instances.
      _class_lock: A class-level lock for thread-safe instance creation.
  """

  # class vars
  _instances: ClassVar[Dict[str, 'CachedBatchMessages']] = {}
  _class_lock: ClassVar[Lock] = Lock()

  # instance vars
  _lock: threading.Lock
  _sight_id: str
  _thread_id: Optional[str]
  _unique_id: Optional[str]
  # Explicitly declare the type of batch_messages
  batch_messages: Dict[int, DecisionMessage]

  def __new__(
      cls,
      sight_id: str = 'default',
      thread_id: Optional[str] = None,
      unique_id: Optional[str] = None,
  ):
    """Create or retrieve a singleton instance for a specific sight_id and thread_id.

    Args:
        sight_id: A unique identifier for the sight.
        thread_id: An optional identifier for the thread.
        unique_id: An optional unique identifier for the instance.

    Returns:
        The singleton instance for the given configuration.
    """
    with cls._class_lock:
      # Create a unique key combining sight_id and thread_id
      instance_key = cls._create_instance_key(sight_id, thread_id, unique_id)

      # If no instance exists for this key, create a new one
      if instance_key not in cls._instances:
        instance = super().__new__(cls)
        # Initialize the instance's attributes
        instance.batch_messages: Dict[int, DecisionMessage] = {}
        instance._lock = Lock()
        instance._sight_id = sight_id
        instance._thread_id = thread_id
        instance._unique_id = unique_id

        # Store the instance in the class-level dictionary
        cls._instances[instance_key] = instance

      return cls._instances[instance_key]

  @classmethod
  def _create_instance_key(
      cls,
      sight_id: str,
      thread_id: Optional[str] = None,
      unique_id: Optional[str] = None,
  ) -> str:
    """Create a unique key for the singleton instance.

    Args:
        sight_id: The primary identifier.
        thread_id: An optional thread identifier.
        unique_id: An optional unique identifier.

    Returns:
        A unique string key representing the instance configuration.
    """
    return f'{sight_id}_{thread_id}_{unique_id}'

  def all_messages(self) -> Dict[int, DecisionMessage]:
    """Return all messages in the cache.

    Returns:
        A dictionary of all cached messages.
    """
    with self._lock:
      return self.batch_messages.copy()

  def get(self, key: int) -> Optional[DecisionMessage]:
    """Retrieve a value from the cache.

    Args:
        key: The key of the message to retrieve.

    Returns:
        The DecisionMessage if found, None otherwise.
    """
    with self._lock:
      return self.batch_messages.get(key)

  def set(self, key: int, value: DecisionMessage) -> None:
    """Set a value in the cache.

    Args:
        key: The key to set.
        value: The DecisionMessage to store.
    """
    with self._lock:
      self.batch_messages[key] = value

  def update(self, key: int, **kwargs) -> None:
    """Update fields of an existing DecisionMessage in the cache.

    Args:
        key: The key of the message to update.
        **kwargs: Keyword arguments representing fields to update.

    Raises:
        KeyError: If the key is not found in the cache.
        AttributeError: If an invalid field is provided.
    """
    with self._lock:
      decision_message = self.batch_messages.get(key)
      if not decision_message:
        raise KeyError(f'Key {key} not found in cache.')

      for field_name, field_value in kwargs.items():
        if hasattr(decision_message, field_name):
          setattr(decision_message, field_name, field_value)
        else:
          raise AttributeError(
              f'Field {field_name} does not exist in DecisionMessage.')

      # Save the updated object back to the cache
      self.batch_messages[key] = decision_message

  def delete(self, key: int) -> None:
    """Delete a value from the cache.

    Args:
        key: The key of the message to delete.
    """
    with self._lock:
      if key in self.batch_messages:
        del self.batch_messages[key]

  def clear(self) -> None:
    """Clear the entire cache."""
    with self._lock:
      self.batch_messages.clear()

  @classmethod
  def get_instance(
      cls,
      sight_id: str = 'default',
      thread_id: Optional[str] = None,
      unique_id: Optional[str] = None,
  ) -> 'CachedBatchMessages':
    """Class method to retrieve an instance by sight_id and thread_id.

    Args:
        sight_id: A unique identifier for the sight.
        thread_id: An optional identifier for the thread.
        unique_id: An optional unique identifier for the instance.

    Returns:
        The singleton instance for the given configuration.
    """
    return cls(sight_id, thread_id, unique_id)

  @property
  def config(self) -> Dict[str, Any]:
    """Retrieve the instance configuration.

    Returns:
        A dictionary of configuration parameters.
    """
    return {
        'sight_id': self._sight_id,
        'thread_id': self._thread_id,
        'unique_id': self._unique_id,
    }
