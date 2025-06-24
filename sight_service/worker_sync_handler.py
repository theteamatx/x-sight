import logging
from sight_service.worker_sync_context import WorkerSyncContext
from sight_service.proto import service_pb2
from sight.utils.proto_conversion import convert_dict_to_proto
from sight.utils.proto_conversion import convert_proto_to_dict
from sight_service.single_action_optimizer import MessageDetails
from google.protobuf import text_format




def handle_propose_action(
    context: WorkerSyncContext, request: service_pb2.ProposeActionRequest
) -> service_pb2.ProposeActionResponse:
  # print('request in propose actions: ', request)

  attributes = convert_proto_to_dict(proto=request.attributes)
  action_attrs = convert_proto_to_dict(proto=request.action_attrs)

  message = MessageDetails.create(action=action_attrs, attributes=attributes)

  unique_id = context.queue.push_message(message)

  response = service_pb2.ProposeActionResponse(action_id=unique_id)
  # logging.info("self.queue => %s", self.queue)
  return response

def handle_finalize_episode(
    context: WorkerSyncContext, request: service_pb2.FinalizeEpisodeRequest
) -> service_pb2.FinalizeEpisodeResponse:
  cache_transport = self.cache_transport

  # If the decision outcome was communicated via the cache.
  if request.decision_messages_ref_key:
    proto_cached_data = cache_transport.fetch_payload(
        request.decision_messages_ref_key)
    text_format.Parse(proto_cached_data, request)

  decision_messages = request.decision_messages
  logging.debug('we have decision messages %s', len(decision_messages))

  for i in range(len(decision_messages)):
    logging.debug('calling queue.complete_message for %s th msg', i)
    context.queue.complete_message(
        worker_id=request.worker_id,
        message_id=decision_messages[i].action_id,
        update_fn=lambda msg: msg.update(
            reward=decision_messages[i].decision_outcome.reward,
            outcome=convert_proto_to_dict(proto=decision_messages[i].
                                          decision_outcome.outcome_params),
            action=convert_proto_to_dict(proto=decision_messages[i].
                                          decision_point.choice_params)))
  return service_pb2.FinalizeEpisodeResponse(response_str='Success!')

def handle_worker_alive(
    context: WorkerSyncContext, request: service_pb2.WorkerAliveRequest
) -> service_pb2.WorkerAliveResponse:
  response = service_pb2.WorkerAliveResponse()

  #! need to add exp_complete field in worker alive request of last iteration
  # if (self.exp_completed):
  #   worker_alive_status = service_pb2.WorkerAliveResponse.StatusType.ST_DONE
  if (not context.queue.get_status()["pending"]):
    worker_alive_status = service_pb2.WorkerAliveResponse.StatusType.ST_RETRY
  else:
    worker_alive_status = service_pb2.WorkerAliveResponse.StatusType.ST_ACT
    batched_msgs = context.queue.create_active_batch(worker_id=request.worker_id)
    for action_id, msg in batched_msgs.items():
      decision_message = response.decision_messages.add()
      decision_message.action_id = action_id
      decision_message.action.CopyFrom(convert_dict_to_proto(dict=msg.action))

  response.status_type = worker_alive_status
  logging.info("worker_alive_status is %s", worker_alive_status)
  return response
