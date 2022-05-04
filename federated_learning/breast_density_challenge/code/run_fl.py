# Copyright (c) 2021, NVIDIA CORPORATION.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import os
import time

from nvflare.fuel.hci.client.fl_admin_api_runner import FLAdminAPIRunner, api_command_wrapper, wait_until_clients_greater_than_cb
from nvflare.fuel.hci.client.fl_admin_api_spec import TargetType


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_number", type=int, default=100, help="FL run number to start at.")
    parser.add_argument("--admin_dir", type=str, default="./admin/", help="Path to admin directory.")
    parser.add_argument("--username", type=str, default="admin@nvflare.com", help="Admin username")
    parser.add_argument("--app", type=str, default="cifar10_fedavg", help="App to be deployed")
    parser.add_argument("--port", type=int, default=8003, help="The admin server port")
    parser.add_argument("--poc", action='store_true', help="Whether admin uses POC mode.")
    parser.add_argument("--min_clients", type=int, default=8, help="Minimum number of clients.")
    args = parser.parse_args()

    host = ""
    port = args.port

    assert os.path.isdir(args.admin_dir), f"admin directory does not exist at {args.admin_dir}"

    # Set up certificate names and admin folders
    upload_dir = os.path.join(args.admin_dir, "transfer")
    if not os.path.isdir(upload_dir):
        os.makedirs(upload_dir)
    download_dir = os.path.join(args.admin_dir, "download")
    if not os.path.isdir(download_dir):
        os.makedirs(download_dir)

    run_number = args.run_number

    # Initialize the runner
    runner = FLAdminAPIRunner(
        host=host,
        port=port,
        username=args.username,
        admin_dir=args.admin_dir,
        poc=args.poc,
        debug=False,
    )

    # Run
    start = time.time()
    # Wait for clients to be connected
    print(f"WAITING FOR {args.min_clients} CLIENTS TO CONNECT...")
    api_command_wrapper(
        runner.api.wait_until_server_status(
            callback=wait_until_clients_greater_than_cb, min_clients=args.min_clients
        )
    )
    print("MAKING SURE CLIENTS ARE READY...")
    time.sleep(30)  # make sure clients are ready

    # Run Training
    print("RUN TRAINING...")
    runner.run(run_number, args.app, restart_all_first=False, shutdown_on_error=False, shutdown_at_end=False,
               timeout=None, min_clients=args.min_clients)
    print("Total training time", time.time() - start)

    # Move client logs to server
    print("GET CLIENT LOGS")
    for client_id in ["site-1", "site-2", "site-3"]:
        result = runner.api.cat_target(target="site-1", file="log.txt")
        if result["status"] == "SUCCESS":
            if "message" in result["details"]:
                log = result["details"]["message"]
                client_log_file = os.path.join(args.admin_dir, "..", f"{client_id}_log.txt")
                with open(client_log_file, "w") as f:
                    f.write(log)
                    print(f"Wrote {client_id}'s log to {client_log_file}")

    print("SHUTDOWN ALL...")
    api_command_wrapper(runner.api.shutdown(TargetType.ALL))


if __name__ == "__main__":
    main()
