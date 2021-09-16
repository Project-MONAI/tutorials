#!/usr/bin/env bash

n_clients=$1

if test -z "$n_clients"
then
      echo "Please provide the number of clients, e.g. ./run_fl.sh 2"
      exit 1
fi

## Start server and clients ##
${projectpath}/fl_utils/fl_run/start_fl.sh ${n_clients}
echo "Waiting for server and clients to start..."
sleep 30s

## Run FL training ##
echo "Run FL training"
python3 ${projectpath}/fl_utils/fl_run_auto/run_fl.py --nr_clients ${n_clients} --run_number 1 \
  --config "../../../spleen_example" \
  --admin_dir "${projectpath}/fl_workspace/admin" &
