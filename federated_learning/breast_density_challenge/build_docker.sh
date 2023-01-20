#!/usr/bin/env bash

# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

#SERVER_FQDN="localhost"
SERVER_FQDN=$1

if test -z "${SERVER_FQDN}"
then
      echo "Usage: ./build_docker.sh [SERVER_FQDN], e.g. ./build_docker.sh localhost"
      exit 1
fi

NEW_IMAGE=monai-nvflare:latest

DOCKER_BUILDKIT=0  # show command outputs
docker build --network=host -t ${NEW_IMAGE} --build-arg server_fqdn=${SERVER_FQDN} -f Dockerfile .
