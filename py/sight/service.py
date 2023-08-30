# Copyright 2023 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from absl import logging
import grpc
from service.decision import decision_pb2
from service.decision import decision_pb2_grpc
import subprocess
from dotenv import load_dotenv
import os
load_dotenv()
import google.auth.transport.requests
import google.oauth2.id_token

def generate_metadata():
  """Generate metadata to call service with authentication"""

  # hosted server on cloud run
  SIGHT_SERVICE_ADDR = 'sight-service-dq7fdwqgbq-uc.a.run.app'
  # SIGHT_SERVICE_ADDR = 'sight-service-greg-dq7fdwqgbq-uc.a.run.app'
  # working for user account
  with open('service/decision/sight_service.cert', 'rb') as f:
      creds = grpc.ssl_channel_credentials(f.read())
  channel_opts = ()
  channel = grpc.secure_channel('{}:{}'.format(SIGHT_SERVICE_ADDR, 443), creds, channel_opts)
  sight_service = decision_pb2_grpc.SightServiceStub(channel)
  logging.info('##### self.sight_service=%s #####', sight_service)
  
  metadata = []
  if (os.getenv("SERVICE_ACCOUNT") == "True"):
    auth_req = google.auth.transport.requests.Request()
    id_token = google.oauth2.id_token.fetch_id_token(auth_req, 'https://'+SIGHT_SERVICE_ADDR)
  else:
    id_token = subprocess.getoutput("gcloud auth print-identity-token")
  
  # print('id token : ',id_token)
  metadata.append(('authorization', 'Bearer ' + id_token))

  return sight_service, metadata
