#!/usr/bin/env bash
SERVER="server"
echo "STARTING ${CLIENT_NAME}"
./fl_workspace/${SERVER}/startup/start.sh; sleep 30s  # TODO: Is there a better way than sleep?
./run_fl.sh 3 mammo_fedavg 1
