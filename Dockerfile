#grab the nvidia cuda image to enable using gpu architecture for container
FROM nvidia/cuda:10.2-cudnn8-runtime-ubuntu18.04

#update and python installation
RUN apt-get update && \
    apt-get install -y python3 python3-pip sudo

#install the requirements
COPY requirements.txt /tmp/requirements.txt
RUN python3 -m pip install --requirement /tmp/requirements.txt

#copy data to container... kinda link not copy
COPY config.json ./config.json
#COPY Out_files_npy ./Out_files_npy

#copy python scripts to container
COPY ./scripts/Utils.py ./scripts/Utils.py
COPY train.py ./train.py

#run training script
#CMD ["python3", "train.py"]