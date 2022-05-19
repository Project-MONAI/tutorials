#!/usr/bin/env bash
SERVER="server"
echo "FINALIZING ${CLIENT_NAME}"
cp -r ./fl_workspace/${SERVER}/run_1 /result/.
cp ./fl_workspace/${SERVER}/*.txt /result/.
cp ./fl_workspace/*_log.txt /result/.
cp ./fl_workspace/${SERVER}/run_1/cross_site_val/cross_val_results.json /result/predictions.json  # only file required for leaderboard computation
# TODO: might need some more standardization of the result folder
