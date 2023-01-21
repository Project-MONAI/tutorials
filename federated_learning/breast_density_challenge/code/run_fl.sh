#!/usr/bin/env bash

# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# add current folder to PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:${PWD}"
echo "PYTHONPATH is ${PYTHONPATH}"
export PYTHONUNBUFFERED=1

algorithms_dir="${PWD}/configs"
workspace="fl_workspace"
admin_username="admin@nvflare.com"
site_pre="site-"

n_clients=$1
config=$2
run=$3

if test -z "${n_clients}" || test -z "${config}" || test -z "${run}"
then
      echo "Usage: ./run_fl.sh [n_clients] [config] [run], e.g. ./run_fl.sh 3 mammo_fedavg 1 0.1"
      exit 1
fi

# start training
echo "STARTING TRAINING"
python3 ./run_fl.py --port=8003 --admin_dir="./${workspace}/${admin_username}" \
  --username="${admin_username}" --run_number="${run}" --app="${algorithms_dir}/${config}" --min_clients="${n_clients}"

# sleep for FL system to shut down, so a new run can be started automatically
sleep 30
echo "TRAINING ENDED"
