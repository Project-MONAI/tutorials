FROM intel/intel-optimized-pytorch:1.11.0-conda

# Update packages
RUN apt update && apt upgrade -y
RUN apt install -y python3-pip python3-venv
RUN conda clean --all

# Create new user
RUN groupadd -g 1000 tami
RUN useradd -rm -d /home/tami -s /bin/bash -g 1000 -u 1000 tami
USER tami
RUN mkdir -p /home/tami/Workspace

# Setup OpenFL and MONAI
RUN mkdir -p /home/tami/Workspace/codes
WORKDIR /home/tami/Workspace/codes
RUN git clone https://github.com/Project-MONAI/tutorials.git
WORKDIR /home/tami/Workspace/codes/tutorials/federated_learning/openfl/openfl_mednist_2d_registration/
RUN python3.8 -m venv director_env
RUN /bin/bash -c 'source director_env/bin/activate && pip3 install -r requirements.txt && deactivate'

WORKDIR /home/tami/Workspace
