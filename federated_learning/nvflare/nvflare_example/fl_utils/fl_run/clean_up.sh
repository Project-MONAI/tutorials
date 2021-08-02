#!/usr/bin/env bash

n_clients=$1

if test -z "$n_clients"
then
      echo "Please provide the number of clients, e.g. ./clean_up.sh 2"
      exit 1
fi

rm -rf ${projectpath}/fl_workspace/server/run_*
rm -rf ${projectpath}/fl_workspace/server/transfer/*
rm ${projectpath}/fl_workspace/server/*.*
for id in $(eval echo "{1..$n_clients}")
do
    rm -rf ${projectpath}/fl_workspace/client${id}/run_*
    rm -rf ${projectpath}/fl_workspace/client${id}/transfer/*
    rm ${projectpath}/fl_workspace/client${id}/*.*
done
