#!/usr/bin/env bash

n_clients=$1

run_number=1

for i in $(eval echo "{1..$n_clients}")
do
    echo "Deploying train config for client${i}"
    cp ${projectpath}/fl_workspace/admin/transfer/spleen_example/config/config_train_${i}.json \
        ${projectpath}/fl_workspace/client${i}/run_${run_number}/mmar_client${i}/config/config_train.json
done
