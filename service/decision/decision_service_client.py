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

"""The Python implementation of the GRPC decision.SightService client."""

from __future__ import print_function

import argparse
import logging
import subprocess
import grpc
from service.decision import decision_pb2
from service.decision import decision_pb2_grpc
from dotenv import load_dotenv
import os
load_dotenv()

import google.auth.transport.requests
import google.oauth2.id_token


# created from reference : https://github.com/GoogleCloudPlatform/python-docs-samples/blob/main/endpoints/bookstore-grpc/bookstore_client.py#L68
def run(host, port, id_token, timeout, use_tls, ca_path):

    with open(ca_path, 'rb') as f:
        creds = grpc.ssl_channel_credentials(f.read())
    print(creds)
    channel_opts = ()

    channel = grpc.secure_channel('{}:{}'.format(host, port), creds, channel_opts)
    # print("channel : ", channel)
    # channel = grpc.insecure_channel('{}:{}'.format(host, port))
    stub = decision_pb2_grpc.SightServiceStub(channel)
    # print("stub : ", stub)

    metadata = []
    print(id_token)
    if id_token:
        metadata.append(('authorization', 'Bearer ' + id_token))

    # print("metadata : ", metadata)
    req = decision_pb2.TestRequest()
    response = stub.Test(req, metadata=metadata)
    print("TestResponse: " + "This testing..." + str(response.val))

    req = decision_pb2.CreateRequest(
                                    log_owner='user@domain.com',
                                    label='bond007',
                                    log_dir_path='/tmp/', 
                                    format='LF_AVRO'
                                )
    response = stub.Create(req,  timeout, metadata=metadata)
    print("CreateResponse: " + "unique id is " + str(response.id) + " and path prefix is : " + response.path_prefix)         

if __name__ == '__main__':

    logging.basicConfig()
    # parser = argparse.ArgumentParser(
    #     description=__doc__,
    #     formatter_class=argparse.RawDescriptionHelpFormatter)
    # parser.add_argument(
    #     '--host', default='localhost', help='The host to connect to')
    # parser.add_argument(
    #     '--port', type=int, default=8080, help='The port to connect to')
    # parser.add_argument(
    #     '--timeout', type=int, default=10, help='The call timeout, in seconds')
    # parser.add_argument(
    #     '--ca_path', type=str, default="service/decision/decision.app", help='The path to the CA.')
    # parser.add_argument(
    #     '--id_token', default=None,
    #     help='The JWT auth token to use for the call')
    # parser.add_argument(
    #     '--use_tls', type=bool, default=False,
    #     help='Enable when the server requires TLS')
    # args = parser.parse_args()

    # service URL from cloud-run except protocol identifier(https://)
    host = "xyz-dq7fdwqgbq-uc.a.run.app"
    port = 443
    timeout = 3000
    use_tls = "True"
    # (downloaded manually from cloud-run website)
    # ca_path = "service/decision/decision.app"
    ca_path = "service/decision/gcert_root.cert"

    # auth token generated from : gcloud auth print-identity-token
    id_token = subprocess.getoutput("gcloud auth print-identity-token")
    # auth_req = google.auth.transport.requests.Request()
    # id_token = google.oauth2.id_token.fetch_id_token(auth_req, 'https://sight-service-dq7fdwqgbq-uc.a.run.app')
    # print('token : '+id_token)

    # run(args.host, args.port, id_token, args.timeout, args.use_tls, args.ca_path)
    # run(args.host, args.port, args.id_token, args.timeout, args.use_tls, args.ca_path)
    run(host, port, id_token, timeout, use_tls, ca_path)
