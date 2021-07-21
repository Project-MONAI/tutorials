#!/usr/bin/env bash

n_clients=$1

if test -z "$n_clients"
then
      echo "Please provide the number of clients, e.g. ./deploy_train_configs.sh 2"
      exit 1
fi

run_number=1

for i in $(eval echo "{1..$n_clients}")
do
    echo "Deploying train config for client${i}"
    cp ${projectpath}/spleen_example/config/config_train_${i}.json \
        ${projectpath}/fl_workspace/client${i}/run_${run_number}/mmar_client${i}/config/config_train.json
done
