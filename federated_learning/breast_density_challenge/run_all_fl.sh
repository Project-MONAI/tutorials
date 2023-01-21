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

mkdir logs
./run_docker_server.sh 2>&1 | tee logs/server_log.txt &
sleep 30s
./run_docker_site-1.sh 0 2>&1 | tee logs/site-1_log.txt &
./run_docker_site-2.sh 1 2>&1 | tee logs/site-2_log.txt &
./run_docker_site-3.sh 0 2>&1 | tee logs/site-3_log.txt
