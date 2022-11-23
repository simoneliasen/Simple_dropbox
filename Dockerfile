FROM nvidia/cuda:8.0-cudnn5-devel-ubuntu16.04

ENV PATH=/opt/conda/bin:$PATH

WORKDIR /simple

COPY . .

RUN apt-get update -y && \
    apt-get upgrade -y && \
    apt-get install -y wget && \
    wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
    bash ~/miniconda.sh -b -p /opt/conda && \
    rm ~/miniconda.sh && \
    conda init

RUN conda env update -n base -f ./environment.yml
