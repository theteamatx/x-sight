"""Logs the state of messages to GCS."""

import abc
import datetime
import queue
import threading
import time

from google.cloud import exceptions
from helpers.logs.logs_handler import logger as logging
from overrides import overrides
from sight_service.message_queue.message_logger.interface import (
    ILogStorageCollectStrategy
)

NotFound = exceptions.NotFound
abstractmethod = abc.abstractmethod
ABC = abc.ABC
Queue = queue.Queue
datetime = datetime.datetime


class MessageFlowLogger:
  """Logs the state of messages to GCS."""

  def __init__(
      self,
      storage_strategy: ILogStorageCollectStrategy,
      chunk_size=500,
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
                        message_details=None,
                        **kwargs):
    """Logs the state of a message to the log buffer.

    Args:
      state: The state of the message.
      message_id: The ID of the message.
      worker_id: The ID of the worker processing the message.
      message_details: Additional details about the message.
      **kwargs : keyword arguments
    """
    log_entry = {
        "state": state,
        "message_id": message_id,
        "worker_id": worker_id,
        "message_details": message_details,
        "timestamp": datetime.utcnow().isoformat(),
        **kwargs
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
