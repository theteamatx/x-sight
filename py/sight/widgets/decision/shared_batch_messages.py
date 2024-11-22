from dataclasses import dataclass
from dataclasses import field
from typing import Any, Dict, Optional


@dataclass
class DecisionMessage:
  action_params: dict
  action_id: int
  reward: Optional[int] = None
  discount: Optional[int] = None
  outcome_params: Optional[dict] = None


class CachedBatchMessages:
  _instance = None  # Class-level variable for the singleton instance

  def __new__(cls, *args, **kwargs):
    if cls._instance is None:
      cls._instance = super(CachedBatchMessages, cls).__new__(cls)
      cls._instance.batch_messages: Dict[int, DecisionMessage] = {
      }  # Initialize the storage
    return cls._instance

  def all_messages(self) -> Dict[int, DecisionMessage]:
    return self.batch_messages

  def get(self, key: int) -> DecisionMessage:
    """Retrieve a value from the cache."""
    return self.batch_messages.get(key)

  def set(self, key: int, value: DecisionMessage) -> None:
    """Set a value in the cache."""
    self.batch_messages[key] = value

  def update(self, key: int, **kwargs) -> None:
    """Update fields of an existing DecisionMessage in the cache."""
    decision_message = self.batch_messages.get(key)
    if not decision_message:
      raise KeyError(f"Key {key} not found in cache.")

    for field_name, field_value in kwargs.items():
      if hasattr(decision_message, field_name):
        setattr(decision_message, field_name, field_value)
      else:
        raise AttributeError(
            f"Field {field_name} does not exist in DecisionMessage.")

    # Save the updated object back to the cache
    self.batch_messages[key] = decision_message

  def delete(self, key: int) -> None:
    """Delete a value from the cache."""
    if key in self.batch_messages:
      del self.batch_messages[key]

  def clear(self) -> None:
    """Clear the entire cache."""
    self.batch_messages.clear()
