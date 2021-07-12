#!/usr/bin/env bash
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

# docker run script for FL admin
# to use host network, use line below
#NETARG="--net=host"
# Admin clients do not need to open ports, so the following line is not needed.
#NETARG="-p 8003:8003"
DOCKER_IMAGE=nvcr.io/nvidia/clara-train-sdk:v4.0
echo "Starting docker with $DOCKER_IMAGE"
docker run --rm -it --name=fladmin -v $DIR/..:/workspace/ -w /workspace/ --ipc=host $NETARG $DOCKER_IMAGE /bin/bash
