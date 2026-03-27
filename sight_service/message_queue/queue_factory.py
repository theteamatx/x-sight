"""Factory for creating message queues."""

from sight_service.message_queue.list_lock_queue import IncrementalUUID
from sight_service.message_queue.list_lock_queue import ListLockMessageQueue
from sight_service.message_queue.list_shared_lock_queue import (
    ListSharedLockMessageQueue
)
from sight_service.message_queue.message_logger.interface import (
    ILogStorageCollectStrategy
)
from sight_service.message_queue.mq_interface import IUUIDStrategy

from .message_logger.log_storage_collect import (
    CachedBasedLogStorageCollectStrategy
)


def queue_factory(
    queue_type: str = 'list',
    batch_size=1,
    logger_storage_strategy:
    ILogStorageCollectStrategy = CachedBasedLogStorageCollectStrategy(),
    id_generator: IUUIDStrategy = IncrementalUUID(),
    **kwargs,
):
  """Factory for creating message queues.

  Args:
    queue_type: The type of queue to create.
    batch_size: The batch size of the queue.
    logger_storage_strategy: The strategy for storing logs.
    id_generator: The strategy for generating UUIDs.
    **kwargs: Additional keyword arguments.

  Returns:
    The created message queue.
  """
  if queue_type == 'list':
    return ListLockMessageQueue(
        id_generator=id_generator,
        batch_size=batch_size,
        logger_storage_strategy=logger_storage_strategy,
    )
  elif queue_type == 'shared_lock_list':
    return ListSharedLockMessageQueue(
        id_generator=id_generator,
        batch_size=batch_size,
        logger_storage_strategy=logger_storage_strategy,
    )
  else:
    return ListSharedLockMessageQueue(
        id_generator=id_generator,
        batch_size=batch_size,
        logger_storage_strategy=logger_storage_strategy,
    )
