#!/usr/bin/env bash
DOCKER_IMAGE=monai-nvflare:latest

GPU=$1
CLIENT_NAME="site-1"

DATA_DIR="${PWD}/data"

# interactive session
#COMMAND="/bin/bash"
# test learner
COMMAND="python3 pt/learners/mammo_learner.py"

echo "Starting $DOCKER_IMAGE with GPU=${GPU}"
docker run -it \
--gpus="device=${GPU}" --network=host --ipc=host --rm --shm-size=1g --ulimit memlock=-1 --ulimit stack=67108864 \
--name="${CLIENT_NAME}_debug" \
-e NVIDIA_VISIBLE_DEVICES=${GPU} \
-v ${DATA_DIR}:/data:ro \
-w /code \
${DOCKER_IMAGE} /bin/bash -c "${COMMAND}"
