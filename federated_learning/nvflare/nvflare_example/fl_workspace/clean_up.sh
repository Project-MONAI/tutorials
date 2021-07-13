#!/usr/bin/env bash

n_clients=$1

if test -z "$n_clients"
then
      echo "Please provide the number of clients, e.g. ./clean_up.sh 2"
      exit 1
fi

rm -rf server/run_*
rm -rf server/transfer/*
rm server/*.*
for id in $(eval echo "{1..$n_clients}")
do
    rm -rf client${id}/run_*
    rm client${id}/*.*
done
