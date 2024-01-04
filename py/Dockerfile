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

FROM google/cloud-sdk:latest as cloud_base

USER root
RUN apt-get -y update
RUN apt-get -y install apt-transport-https ca-certificates gnupg
RUN apt-get -y install python3.10

RUN apt-get -y install virtualenv
RUN apt-get -y install python3-pip
RUN apt-get -y install vim

WORKDIR /x-sight
COPY py/sight/requirements.txt py/sight/requirements.txt
RUN pip install -r py/sight/requirements.txt

COPY py py
COPY sight_service sight_service
COPY .env .env

ENV PYTHONPATH "$HOME/x-sight:$HOME/x-sight/py:${PYTHONPATH}"
