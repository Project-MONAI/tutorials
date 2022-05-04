#!/usr/bin/env bash
DOCKER_IMAGE=monai-nvflare:latest

GPU=$1
CLIENT_NAME="site-2"

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
