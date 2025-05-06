# Dockerfile may have following Arguments:
# tag - tag for the Base image, (e.g. 2.9.1 for tensorflow)
# branch - user repository branch to clone (default: master, another option: test)
#
# To build the image:
# $ docker build -t <dockerhub_user>/<dockerhub_repo> --build-arg arg=value .
# or using default args:
# $ docker build -t <dockerhub_user>/<dockerhub_repo> .
#
# [!] Note: For the Jenkins CI/CD pipeline, input args are defined inside
# the Jenkinsfile, not here!

ARG tag=2.10.0

# Base image, e.g. tensorflow/tensorflow:2.9.1
FROM tensorflow/tensorflow:${tag}-gpu

LABEL maintainer='Fahimeh Alibabaei'
LABEL version='0.0.1'
# Deepaas API for TBBRDet Model

# What user branch to clone [!]
ARG branch=nvflare

# Install Ubuntu packages / applications
RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y git && \
    apt-get install -y libgl1 && \
    apt-get install -y wget && \
    rm -rf /var/lib/apt/lists/*

# Set LANG environment
ENV LANG C.UTF-8

# Set the working directory
WORKDIR /srv

# Install rclone (needed if syncing with NextCloud for training; otherwise remove)
RUN curl -O https://downloads.rclone.org/rclone-current-linux-amd64.deb && \
    dpkg -i rclone-current-linux-amd64.deb && \
    apt install -f && \
    mkdir /srv/.rclone/ && \
    touch /srv/.rclone/rclone.conf && \
    rm rclone-current-linux-amd64.deb && \
    rm -rf /var/lib/apt/lists/*

ENV RCLONE_CONFIG=/srv/.rclone/rclone.conf

# Initialization scripts
# deep-start can install JupyterLab or VSCode if requested
RUN git clone https://github.com/ai4os/deep-start /srv/.deep-start && \
    ln -s /srv/.deep-start/deep-start.sh /usr/local/bin/deep-start

# Necessary for the Jupyter Lab terminal
ENV SHELL /bin/bash

# Check if the CUDA path exists and set CUDA_HOME
RUN cuda_path="/usr/local/cuda/" && \
    if [ -d "$cuda_path" ]; then \
        echo "CUDA_HOME is set to $cuda_path"; \
        echo "export CUDA_HOME=$cuda_path" >> /etc/profile.d/cuda.sh; \
    else \
        echo "Path $cuda_path does not exist! CUDA_HOME cannot be set." && exit 1; \
    fi
# Ensure the CUDA_HOME environment variable is available in the container
ENV CUDA_HOME=/usr/local/cuda/

# Install user app
# make sure to update pip so that installations in editable mode (-e) work!
RUN git clone --depth 1 -b $branch --recurse-submodules https://github.com/ai4os-hub/thermal-urban-feature-segmenter.git && \
    cd thermal-urban-feature-segmenter && \
    git pull --recurse-submodules && \
    git submodule update --remote --recursive && \
    pip3 install -U pip && \
    pip3 install packaging==22.0 && \
    pip3 install --no-cache-dir -e ./TUFSeg && \
    pip3 install --no-cache-dir -e . && \
    pip3 install --no-deps  -r nvflare/requirements_nvflare.txt && \
    cd ..
