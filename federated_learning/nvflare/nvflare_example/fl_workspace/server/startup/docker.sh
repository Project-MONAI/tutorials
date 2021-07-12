#!/usr/bin/env bash
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
# docker run script for FL server
# local data directory
MY_DATA_DIR=/home/data
# to use host network, use line below
NETARG="--net=host"
# or to expose specific ports, use line below
#NETARG="-p 8003:8003 -p 8002:8002"
DOCKER_IMAGE=nvcr.io/nvidia/clara-train-sdk:v4.0
echo "Starting docker with $DOCKER_IMAGE"
docker run --rm -it --name=flserver -v $DIR/..:/workspace/ -v $MY_DATA_DIR:/data/ -w /workspace/ --ipc=host $NETARG $DOCKER_IMAGE /bin/bash
