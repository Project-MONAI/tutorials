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

OUT_DIR="${PWD}/result_server"
SERVER="server"

GPU=$1

COMMAND="/code/start_server.sh; /code/finalize_server.sh"

echo "Starting $DOCKER_IMAGE with GPU=${GPU}"
docker run \
--gpus="device=${GPU}" --network=host --ipc=host --rm --shm-size=1g --ulimit memlock=-1 --ulimit stack=67108864 \
--name=${SERVER} \
-e NVIDIA_VISIBLE_DEVICES="${GPU}" \
-v "${OUT_DIR}":/result \
-w /code \
${DOCKER_IMAGE} /bin/bash -c "${COMMAND}"

# kill client containers
docker kill site-1 site-2 site-3
