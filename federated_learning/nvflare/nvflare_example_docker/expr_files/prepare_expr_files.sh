# set fl docker image name
export FL_DOCKER_IMAGE="monai_nvflare:latest"
DOCKER_PROVISION_PATH=/opt/conda/lib/python3.8/site-packages/provision
DEMO_PROVISION_PATH="expr_files"
provision -p $DEMO_PROVISION_PATH/project.yml -a $DEMO_PROVISION_PATH/authz_config.json -o $DEMO_PROVISION_PATH
cp $DOCKER_PROVISION_PATH/audit.pkl $DEMO_PROVISION_PATH
cd /fl_workspace/; chown -R 1000:1000 *

# if you do not need to download the spleen dataset, please comment the following lines.

# The docker run command in `build_docker_provision.sh` mounts the path of the
# current folder into `/fl_workspace`, thus the downloaded Spleen dataset will be
# in the current folder.
DATASET_DOWNLOAD_PATH="/fl_workspace/"
python expr_files/download_dataset.py -root_dir $DATASET_DOWNLOAD_PATH

# prepare modified data list files, if your Spleen dataset path is different, please
# modify the following line.
cp spleen_example/config/dataset_part*.json $DATASET_DOWNLOAD_PATH/Task09_Spleen/
