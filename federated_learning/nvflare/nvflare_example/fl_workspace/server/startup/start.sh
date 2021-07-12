#!/usr/bin/env bash
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
export CUDA_VISIBLE_DEVICES=
echo "WORKSPACE set to $DIR/.."
mkdir -p $DIR/../transfer 
$DIR/sub_start.sh &
