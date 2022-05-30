#!/usr/bin/env bash

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
