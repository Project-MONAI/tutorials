#!/usr/bin/env python3
import os
import argparse
import time
import re
import sys
import json
import shutil
import uuid
# use source
#src_root = "/workspace/Code/clara4.0"
#python_paths = [f"{src_root}/common/dlmed/src",
#                f"{src_root}/train/automl/src",
#                f"{src_root}/flare"]
#[sys.path.insert(0, item) for item in python_paths]

from dlmed.hci.client.api import AdminAPI
from api_utils import api_command_wrapper, wait_to_complete, fl_shutdown


def create_tmp_config_dir(upload_dir, config):
    tmp_config = str(uuid.uuid4())
    print(f"Creating temporary config from {config} -> {tmp_config}")
    tmp_config_dir = os.path.join(upload_dir, tmp_config)  # creat a temp config for this run
    if os.path.isdir(tmp_config_dir):
        shutil.rmtree(tmp_config_dir)
    shutil.copytree(os.path.join(upload_dir, config), tmp_config_dir)

    return tmp_config, tmp_config_dir


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--nr_clients', type=int, default=2, help="Minimum number of clients.")
    parser.add_argument('--run_number', type=int, default=1, help="FL run number.")
    parser.add_argument('--config', type=str, default='spleen_example', help="directory name with training configs.")
    parser.add_argument('--admin_dir', type=str, default='./admin', help="Path to admin directory.")

    args = parser.parse_args()

    host = 'localhost'
    port = 8003

    # Set up certificate names and admin folders
    ca_cert = os.path.join(args.admin_dir, 'startup', 'rootCA.pem')
    client_cert = os.path.join(args.admin_dir, 'startup', 'client.crt')
    client_key = os.path.join(args.admin_dir, 'startup', 'client.key')
    upload_dir = os.path.join(args.admin_dir, 'transfer')
    download_dir = os.path.join(args.admin_dir, 'download')
    if not os.path.isdir(download_dir):
        os.makedirs(download_dir)

    assert os.path.isdir(args.admin_dir), f"admin directory does not exist at {args.admin_dir}"
    assert os.path.isfile(ca_cert), f"rootCA.pem does not exist at {ca_cert}"
    assert os.path.isfile(client_cert), f"client.crt does not exist at {client_cert}"
    assert os.path.isfile(client_key), f"client.key does not exist at {client_key}"

    # Connect with admin client
    api = AdminAPI(
        host=host,
        port=port,
        ca_cert=ca_cert,
        client_cert=client_cert,
        client_key=client_key,
        upload_dir=upload_dir,
        download_dir=download_dir,
        debug=False
    )
    reply = api.login(username="admin@nvidia.com")
    for k in reply.keys():
        assert "error" not in reply[k].lower(), f"Login not successful with {reply}"

    # Execute commands
    api_command_wrapper(api, "set_timeout 30")

    # create a temporary config for editing
    try:
        tmp_config, tmp_config_dir = create_tmp_config_dir(upload_dir, args.config)
    except BaseException as e:
        print(f"There was an exception {e}. Shutting down clients and server.")
        fl_shutdown(api)
        sys.exit(1)

    # update server config to set min_num_clients:
    server_config_file = os.path.join(tmp_config_dir, 'config', 'config_fed_server.json')
    with open(server_config_file, 'r') as f:
        server_config = json.load(f)
    server_config['servers'][0]['min_num_clients'] = args.nr_clients
    with open(server_config_file, 'w') as f:
        json.dump(server_config, f, indent=4)

    api_command_wrapper(api, "check_status server")

    api_command_wrapper(api, f"set_run_number {args.run_number}")

    api_command_wrapper(api, f"upload_folder {tmp_config}")

    api_command_wrapper(api, f"deploy {tmp_config} server")

    api_command_wrapper(api, "start server")

    time.sleep(10)
    # deploy clients
    for client_id in range(1, args.nr_clients+1):
        # update client's train config to set seed:
        ref_train_config_file = os.path.join(upload_dir, args.config, 'config', f'config_train_{client_id}.json')
        train_config_file = os.path.join(tmp_config_dir, 'config', f'config_train.json')

        print(f"Deploying train config for client{client_id}")
        shutil.copyfile(ref_train_config_file, train_config_file)

        # upload & deploy on client
        api_command_wrapper(api, f"upload_folder {tmp_config}")
        api_command_wrapper(api, f"deploy {tmp_config} client client{client_id}")

    api_command_wrapper(api, "start client")

    # delete temporary config
    if os.path.isdir(tmp_config_dir):
        shutil.rmtree(tmp_config_dir)

    # Keep checking the server and client statuses until FL training is complete.
    wait_to_complete(api, interval=30)

    # shutdown
    fl_shutdown(api)

    # log out
    print('Admin logging out.')
    api.logout()


if __name__ == "__main__":
    main()
