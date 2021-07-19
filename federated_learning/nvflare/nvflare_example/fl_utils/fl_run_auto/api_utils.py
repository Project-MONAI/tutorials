import time
import re


def api_command_wrapper(api, command):
    print("\nISSUING COMMAND: {}".format(command))
    reply = api.do_command(command)
    print(reply)
    assert reply['status'] == 'SUCCESS', "command was not successful!"

    # check for other errors
    for r in reply['data']:
        if r['type'] == 'string':
            #  print(r['data'])
            assert 'error' not in r['data'].lower(), f"there was an error in reply executing: {command}"
        if r['type'] == 'error':
            raise ValueError(f"there was an error executing: {command}")
    return reply


def wait_to_complete(api, interval=60):
    fl_is_training = True
    while fl_is_training:
        time.sleep(interval)
        reply = api_command_wrapper(api, "check_status client")
        nr_clients_starting = len([m.start() for m in re.finditer('status: training starting', reply['data'][0]['data'])])
        nr_clients_started = len([m.start() for m in re.finditer('status: training started', reply['data'][0]['data'])])
        nr_clients_crosssiteval = len([m.start() for m in re.finditer('status: cross site validation', reply['data'][0]['data'])])
        print(f'{nr_clients_starting} clients starting training.')
        print(f'{nr_clients_started} clients in training.')
        print(f'{nr_clients_crosssiteval} clients in cross-site validation.')

        reply = api_command_wrapper(api, "check_status server")
        if 'status: training stopped' in reply['data'][0]['data']:
            server_is_training = False
            print('Server stopped.')
        else:
            print('Server is training.')
            server_is_training = True

        if (~server_is_training) and \
           (nr_clients_started == 0) and \
           (nr_clients_starting == 0) and \
           (nr_clients_crosssiteval == 0):
            fl_is_training = False
            print('FL training & cross-site validation stopped/completed.')

    return True


def fl_shutdown(api):
    print('Shutting down FL system...')
    api_command_wrapper(api, "shutdown client")
    time.sleep(10)
    api_command_wrapper(api, "shutdown server")
