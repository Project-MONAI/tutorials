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

SERVER="server"
echo "FINALIZING ${CLIENT_NAME}"
cp -r ./fl_workspace/${SERVER}/run_1 /result/.
cp ./fl_workspace/${SERVER}/*.txt /result/.
cp ./fl_workspace/*_log.txt /result/.
cp ./fl_workspace/${SERVER}/run_1/cross_site_val/cross_val_results.json /result/predictions.json  # only file required for leaderboard computation
# TODO: might need some more standardization of the result folder
