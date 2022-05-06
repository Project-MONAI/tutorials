#!/usr/bin/env bash
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
