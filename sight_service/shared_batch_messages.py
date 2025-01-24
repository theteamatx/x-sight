"""A shared batch messages module"""

import dataclasses
import threading
from typing import Any, Dict, Optional

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
  """A class for caching batch messages.

  Attributes:
      batch_messages: A dictionary of cached DecisionMessages.
  """

  def __init__(self):
    """Initialize the CachedBatchMessages instance."""
    self._lock = threading.Lock()
    self.batch_messages: Dict[int, DecisionMessage] = {}

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
