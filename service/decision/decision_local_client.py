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

import logging

import grpc
from service.decision import decision_pb2
from service.decision import decision_pb2_grpc


def run():
    with grpc.insecure_channel('localhost:8080') as channel:
        stub = decision_pb2_grpc.SightServiceStub(channel)

        response = stub.Test(decision_pb2.TestRequest())
        print("TestResponse: " + "unique id is " + str(response.val))
        
        response = stub.Create(decision_pb2.CreateRequest(
                                    log_owner='user@domain.com',
                                    label='dummy007',
                                    log_dir_path='/tmp/', 
                                    format='LF_AVRO'
                                ))
        print("CreateResponse: " + "unique id is " + str(response.id) + " and path prefix is : " + response.path_prefix)
        
        response = stub.CreateStudy(decision_pb2.StudyRequest(
                                    vizier_endpoint = 'us-central1-aiplatform.googleapis.com',
                                    parent = 'projects/cameltrain/locations/us-central1'
                                ))
        print("CreateStudyResponse: " + "vizier_study is :" + str(response.vizier_study))
        
if __name__ == '__main__':
    logging.basicConfig()
    run()
