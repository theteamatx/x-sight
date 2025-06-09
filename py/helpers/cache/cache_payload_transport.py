"""Cached Payload Transport."""

import json
from typing import Any, Optional
import uuid

from .cache_interface import CacheInterface


class CachedPayloadTransport:
  """Transport for storing and retrieving payloads using Redis.

  Payloads are stored with a time-to-live (TTL) and can be optionally deleted
  after retrieval.

  TTL : (time-to-live only works with the in-memory databases)

  """

  def __init__(self, cache: CacheInterface, ttl_seconds: int = 300):
    """Initializes the CachedPayloadTransport.

    Args:
        cache : This is a actual cache which may be (local, redis, gcs, or any other side)
        ttl_seconds: Time-to-live for cached payloads (default: 5 minutes) (This is for in-memory databases only)
    """
    self.cache = cache
    self.ttl = ttl_seconds

  def _generate_key(self) -> str:
    """Generate a secure, unique reference key."""
    return f"payload:{uuid.uuid4()}"

  def store_payload(self, data: Any) -> str:
    """Stores the data in Redis and returns a reference key.

    Args:
        data: Any JSON-serializable data
    Returns:
        The reference key for the stored payload.
    """
    key = self._generate_key()
    self.cache.set(key=key, value=data)
    return key

  def fetch_payload(self, key: str, delete_after: bool = True) -> Optional[Any]:
    """Retrieves the data from Redis using the reference key.

    Args:
        key: The reference key returned by store_payload
        delete_after: Whether to delete the payload after retrieval
    Returns:
        The original payload or None if not found
    """
    value = self.cache.get(key)
    if value is None:
      return None

    # if delete_after:
    #   self.cache.delete(key)

    return value
