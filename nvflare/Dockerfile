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
FROM ai4oshub/thermal-urban-feature-segmenter

LABEL maintainer='Elena Vollmer'
LABEL version='0.0.1'
# Deepaas API for TBBRDet Model

# What user branch to clone [!]
ARG branch=nvflare

# Install Ubuntu packages / applications

# Install user app
# make sure to update pip so that installations in editable mode (-e) work!
RUN cd thermal-urban-feature-segmenter && \
    git remote set-branches origin nvflare && \
    git fetch --depth=1 origin nvflare && \
    git checkout nvflare && \
    git submodule update --remote --recursive && \
    pip3 install nvflare==2.5.0 perun && \
    pip3 install --no-deps protobuf==3.20
