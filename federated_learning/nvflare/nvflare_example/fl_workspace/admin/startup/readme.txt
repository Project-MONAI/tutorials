*********************************
Admin Client package
*********************************
The package includes the following files:
readme.txt
rootCA.pem
client.crt
client.key
fl_admin.sh
fl_admin.bat
clara_hci-3.1.0-py3-none-any.whl

Please install the clara_hci-3.1.0-py3-none-any.whl file by 'python3 -m pip install.'  This will install a set of Python codes
in your environment.  After installation, you can run the fl_admin.sh or fl_admin.bat file to start communicating to the admin server.

The rootCA.pem file is pointed by "ca_cert" in fl_admin.sh/fl_admin.bat.  If you plan to move/copy it to a different place,
you will need to modify fl_admin.sh.  The same applies to the other two files, client.crt and client.key.

The email in your submission to participate this Federated Learning project is embedded in the CN field of client
 certificate, which uniquely identifies the participant.  As such, please safeguard its private key, client.key.
