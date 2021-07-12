#!/usr/bin/env bash

n_clients=$1

rm -rf server/run_*
rm -rf server/transfer/*
rm server/*.*
for id in $(eval echo "{1..$n_clients}")
do
    rm -rf client${id}/run_*
    rm client${id}/*.*
done
