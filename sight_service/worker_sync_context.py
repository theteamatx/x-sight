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
from sight.utils.proto_conversion import convert_dict_to_proto
from sight_service.proto import service_pb2
from sight_service.single_action_optimizer import MessageDetails


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
  is_exp_completed: Optional[str] = None

  def add_outcome_to_outcome_response(
      self, msg_details: MessageDetails, sample_id,
      outcome: service_pb2.GetOutcomeResponse.Outcome):
    outcome.action_id = sample_id
    outcome.status = service_pb2.GetOutcomeResponse.Outcome.Status.COMPLETED
    outcome.reward = msg_details.reward
    outcome.action_attrs.CopyFrom(
        convert_dict_to_proto(dict=msg_details.action))
    outcome.outcome_attrs.CopyFrom(
        convert_dict_to_proto(dict=msg_details.outcome))
    outcome.attributes.CopyFrom(
        convert_dict_to_proto(dict=msg_details.attributes))
