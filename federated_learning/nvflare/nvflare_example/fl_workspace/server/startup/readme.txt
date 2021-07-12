*********************************
Federated Learning Server package
*********************************
The package includes the following files:
readme.txt
rootCA.pem
server.crt
server.key
fed_server.json
start.sh

Run start.sh to start the server.

The rootCA.pem file is pointed by "ssl_root_cert" in fed_server.json.  If you plan to move/copy it to a different place,
you will need to modify fed_server.json.  The same applies to the other two files, server.crt and server.key.

Please always safeguard the server.key.
