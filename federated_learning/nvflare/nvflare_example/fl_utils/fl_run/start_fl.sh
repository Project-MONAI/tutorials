#!/usr/bin/env bash

n_clients=$1

if test -z "$n_clients"
then
      echo "Please provide the number of clients, e.g. ./start_fl.sh 2"
      exit 1
fi

# Start server
echo "Starting server and ${n_clients} clients"
${projectpath}/fl_workspace/server/startup/start.sh
sleep 10s

# Start clients
gpu_idx=0
for i in $(eval echo "{1..$n_clients}")
do
    echo "Starting client${i} on GPU ${gpu_idx}"
    export CUDA_VISIBLE_DEVICES=${gpu_idx}
    ${projectpath}/fl_workspace/client${i}/startup/start.sh
    ((gpu_idx=gpu_idx+1))
done
