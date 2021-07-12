#!/usr/bin/env bash
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
mkdir -p $DIR/../transfer
python3 -m dlmed.hci.tools.admin --host localhost --port 8003 --prompt "> " --with_file_transfer --upload_dir=$DIR/../transfer --download_dir=$DIR/../transfer --with_shell --with_login --with_ssl --cred_type cert --ca_cert=$DIR/rootCA.pem --client_cert=$DIR/client.crt --client_key=$DIR/client.key
