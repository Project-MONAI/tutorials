#!/usr/bin/env bash

mkdir logs
./run_docker_server.sh 2>&1 | tee logs/server_log.txt &
sleep 30s
./run_docker_site-1.sh 0 2>&1 | tee logs/site-1_log.txt &
./run_docker_site-2.sh 1 2>&1 | tee logs/site-2_log.txt &
./run_docker_site-3.sh 0 2>&1 | tee logs/site-3_log.txt
