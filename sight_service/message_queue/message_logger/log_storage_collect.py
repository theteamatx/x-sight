"""LogStorageCollectStrategy module."""

from helpers.cache.cache_factory import CacheFactory
from helpers.logs.logs_handler import logger as logging
from overrides import overrides
from sight_service.message_queue.message_logger.interface import (
    ILogStorageCollectStrategy
)


class NoneLogStorageCollectStrategy(ILogStorageCollectStrategy):

  @overrides
  def save_logs(self, logs):
    logging.info('NOT SAVING LOGS FOR MQ')
    pass

  @overrides
  def collect_logs(self) -> list[str]:
    logging.info('NOT COLLECTING LOGS FOR MQ')
    return []


class CachedBasedLogStorageCollectStrategy(ILogStorageCollectStrategy):
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

  @overrides
  def save_logs(self, logs):
    """Saves logs using the configured cache.

    Args:
        logs (list[str]): The list of log entries to save.
    """
    try:
      file_name = f'{self.dir_prefix}chunk_{self.current_file_number}'
      logs = logs if isinstance(logs, list) else [logs]
      logging.debug(f'TRYING TO SET THIS CHUNK {file_name}')
      self.cache.json_set(file_name, logs)  # Store the log chunk
      self.current_file_number += 1
    except (KeyError, ValueError) as e:
      logging.error(f'Failed to save logs: {e}')

  @overrides
  def collect_logs(self) -> list[str]:
    """Collects logs from all available chunks.

    Returns:
        list: A list of all log entries across chunks.
    """
    all_logs = []
    try:
      chunk_files = self.cache.json_list_keys(
          prefix=self.dir_prefix)  # Get all files matching the prefix
      logging.debug(f'TRYING TO COLLECT ALL CHUNKS {len(chunk_files)}')
      for file_name in sorted(
          chunk_files
      ):  # Sort for deterministic order like chunk 0 , chunk 1 , etc..
        logs = self.cache.json_get(f'{file_name}')
        if logs:
          all_logs.extend(logs)
    except (KeyError, ValueError) as e:
      logging.error(f'Failed to collect logs: {e}')
    return all_logs
