*********************************
Federated Learning Client package
*********************************
The package includes the following files:
readme.txt
rootCA.pem
client.crt
client.key
fed_client.json
start.sh

Run start.sh to start the client.

The rootCA.pem file is pointed by "ssl_root_cert" in fed_client.json.  If you plan to move/copy it to a different place,
you will need to modify fed_client.json.  The same applies to the other two files, client.crt and client.key.

The client name in your submission to participate this Federated Learning project is embedded in the CN field of client
 certificate, which uniquely identifies the participant.  As such, please safeguard its private key, client.key.
