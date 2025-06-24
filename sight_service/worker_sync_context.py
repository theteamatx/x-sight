import dataclasses
from dataclasses import field, dataclass
import threading
import datetime
from typing import Any, Dict, Optional
from sight_service.message_queue.mq_interface import IMessageQueue
from sight_service.message_queue.message_logger.interface import (
    ILogStorageCollectStrategy
)
from sight_service.message_queue.message_logger.log_storage_collect import (
    CachedBasedLogStorageCollectStrategy
)
from sight_service.message_queue.queue_factory import queue_factory
datetime = datetime.datetime

logger_storage_strategy: ILogStorageCollectStrategy = (
        CachedBasedLogStorageCollectStrategy(
            cache_type="gcs",
            config={
                "gcs_base_dir": "sight_mq_logs_for_analysis",
                "gcs_bucket": "cameltrain-sight",
                "dir_prefix": f'log_chunks_{datetime.utcnow().strftime("%Y-%m-%d_%H-%M-%S")}/',
            },
        ))

@dataclass
class WorkerSyncContext:
  queue: Optional[IMessageQueue] = None
