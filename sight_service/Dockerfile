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

# syntax=docker/dockerfile:1

# using base image
FROM python:3.10
LABEL Maintainer="bronevet@google.com"

# set up working directory in container
WORKDIR /x-sight

# Installing pip
RUN set -xe \
    && apt-get update -y \
    && apt-get install -y python3-pip

# download necessary dependency in container
COPY sight_service/requirements.txt sight_service/requirements.txt
RUN pip3 install --upgrade pip
RUN pip3 install -r sight_service/requirements.txt

COPY .env .env

# copy all files from proto/ recursively  in container
COPY py/sight/proto/ py/sight/proto/

# copy all files from sight_service/ recursively in container
COPY sight_service/ sight_service/

# updating PYTHONPATH so, imports in server.py code works correctly
ENV PYTHONPATH "$HOME/x-sight:$HOME/x-sight/py:${PYTHONPATH}"

# run the python command
CMD [ "python3", "sight_service/service_root.py"]
