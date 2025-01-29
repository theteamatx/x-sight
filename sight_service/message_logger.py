"""Logs the state of messages to GCS."""

import abc
import datetime
import json
import os
import queue
import threading
import time

from google.cloud import exceptions
from google.cloud import storage
from helpers.cache.cache_factory import CacheFactory
from helpers.logs.logs_handler import logger as logging
from overrides import overrides

NotFound = exceptions.NotFound
abstractmethod = abc.abstractmethod
ABC = abc.ABC
Queue = queue.Queue
datetime = datetime.datetime


class LogStorageCollectStrategyABC(ABC):

  @abstractmethod
  def save_logs(self, logs):
    pass

  @abstractmethod
  def collect_logs(self) -> list:
    pass


class LogStorageCollectStrategyEmpty(LogStorageCollectStrategyABC):

  @overrides
  def save_logs(self, logs):
    logging.info('NOT SAVING LOGS FOR MQ')
    pass

  @overrides
  def collect_logs(self) -> list:
    logging.info('NOT COLLECTING LOGS FOR MQ')
    pass


class LogStorageCollectStrategy(LogStorageCollectStrategyABC):
  """A unified log storage strategy using the Cache module."""

  def __init__(self, cache_type='local', config=None):
    """Initializes the CacheBasedLogStorageStrategy.

    Args:
        cache_type (str): The type of cache to use (e.g., "local", "gcs",
          "none").
        config (dict): Configuration specific to the cache type.
    """
    if config is None:
      config = {}
    self.cache = CacheFactory.get_cache(cache_type, config=config)
    self.current_file_number = 0
    self.dir_prefix = config.get('dir_prefix', 'log_chunks/')

  def save_logs(self, logs):
    """Saves logs using the configured cache.

    Args:
        file_content (str): The content of the log file in JSON format.
    """
    try:
      file_name = f'{self.dir_prefix}chunk_{self.current_file_number}'
      logs = logs if isinstance(logs, list) else [logs]
      self.cache.json_set(file_name, logs)  # Store the log chunk
      self.current_file_number += 1
    except (KeyError, ValueError) as e:
      logging.error(f'Failed to save logs: {e}')

  def collect_logs(self):
    """Collects logs from all available chunks.

    Returns:
        list: A list of all log entries across chunks.
    """
    all_logs = []
    try:
      chunk_files = self.cache.json_list_keys(
          prefix=self.dir_prefix)  # Get all files matching the prefix
      for file_name in sorted(chunk_files):  # Sort for deterministic order
        logs = self.cache.json_get(f'{file_name}')
        if logs:
          all_logs.extend(logs)
    except Exception as e:
      logging.error(f'Failed to collect logs: {e}')
    return all_logs


class MessageFlowLogger:
  """Logs the state of messages to GCS."""

  def __init__(
      self,
      storage_strategy: LogStorageCollectStrategy,
      chunk_size=100,
      flush_interval=5,
  ):
    self.log_buffer = Queue()  # Thread-safe buffer
    self.chunk_size = chunk_size
    self.flush_interval = flush_interval
    self.storage_strategy = storage_strategy  # Inject the strategy
    self.lock = threading.Lock()
    self.stop_event = threading.Event()
    self.worker_thread = threading.Thread(target=self._flush_periodically)
    self.worker_thread.start()

  def log_message_state(self,
                        state,
                        message_id,
                        worker_id=None,
                        message_details=None):
    """Logs the state of a message to the log buffer.

    Args:
      state: The state of the message.
      message_id: The ID of the message.
      worker_id: The ID of the worker processing the message.
      message_details: Additional details about the message.
    """
    log_entry = {
        "state": state,
        "message_id": message_id,
        "worker_id": worker_id,
        "message_details": message_details,
        "timestamp": datetime.utcnow().isoformat(),
    }
    self.log_buffer.put(log_entry)  # Add to buffer
    if self.log_buffer.qsize() >= self.chunk_size:
      self._flush_logs()

  def _flush_logs(self):
    with self.lock:  # Ensure thread safety
      logs_to_flush = []
      while (not self.log_buffer.empty() and
             len(logs_to_flush) < self.chunk_size):
        logs_to_flush.append(self.log_buffer.get())
      if logs_to_flush:
        self._save_logs(logs_to_flush)

  def _save_logs(self, logs):
    self.storage_strategy.save_logs(logs=logs)

  def _flush_periodically(self):
    while not self.stop_event.is_set():
      time.sleep(self.flush_interval)
      self._flush_logs()

  def stop(self):
    self.stop_event.set()
    self.worker_thread.join()
    self._flush_logs()  # Final flush before shutting down
