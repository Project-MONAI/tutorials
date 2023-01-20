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

DOCKER_IMAGE=monai-nvflare:latest

GPU=$1
CLIENT_NAME="site-3"

DATA_DIR="${PWD}/data"

COMMAND="/code/start_${CLIENT_NAME}.sh; tail -f /dev/null"

echo "Starting $DOCKER_IMAGE with GPU=${GPU}"
docker run \
--gpus="device=${GPU}" --network=host --ipc=host --rm --shm-size=1g --ulimit memlock=-1 --ulimit stack=67108864 \
--name="${CLIENT_NAME}" \
-e NVIDIA_VISIBLE_DEVICES="${GPU}" \
-v "${DATA_DIR}":/data:ro \
-w /code \
${DOCKER_IMAGE} /bin/bash -c "${COMMAND}"
