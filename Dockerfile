#grab the nvidia cuda image to enable using gpu architecture for container
FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

#update and python installation
RUN apt-get -qq update && \
    apt-get install -y python3 python3-pip sudo

#install the requirements
COPY requirements.txt /tmp/
RUN pip install --requirement /tmp/requirements.txt

#copy data to container... kinda link not copy
COPY input_98660.npy ./input_98660.npy
COPY Out_files_npy ./Out_files_npy

#copy python scripts to container
COPY ./scripts/Utils.py ./scripts/Utils.py
COPY train.py ./train.py

#run training script
#ENTRYPOINT ['python3', 'train.py']