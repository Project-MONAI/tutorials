#!/bin/bash

DOCKER_IMAGE=monai_nvflare

# check if image exist, if not build it
docker images |grep ${DOCKER_IMAGE}
dockerNameExist=$?
if ((${dockerNameExist}==0)) ;then
  echo --- docker image ${DOCKER_Run_Name} exist
else
  echo --- docker image ${DOCKER_Run_Name} does not exist, building it
  docker build -f docker_files/Dockerfile --tag ${DOCKER_IMAGE} .
  echo ----------- docker image ${DOCKER_Run_Name} built
fi
# using the built docker image to provision
cmd2run="bash expr_files/prepare_expr_files.sh"
docker run \
  --rm \
  --shm-size 1G \
  -v ${PWD}/:/fl_workspace/ \
  -w /fl_workspace/ \
  -it \
  ${DOCKER_IMAGE} \
  ${cmd2run}

echo ------------------ exited from docker image
