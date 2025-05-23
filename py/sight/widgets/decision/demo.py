from sight_service.proto import service_pb2

req = service_pb2.FinalizeEpisodeRequest()

req.decision_messages_ref_key = 'something'
req.client_id = 'first'

from google.protobuf import text_format

text_proto = text_format.MessageToString(req)

import json

print(text_proto)

y = json.dumps(text_proto)

response = service_pb2.FinalizeEpisodeRequest()

response.question_label = 'new_data'

text_format.Parse(json.loads(y), response)

print(text_format.MessageToString(response))
