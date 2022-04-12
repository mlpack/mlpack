FROM ubuntu:latest
ENV PATH="/root/miniconda3/bin:${PATH}"
ARG PATH="/root/miniconda3/bin:${PATH}"
ENV LD_LIBRARY_PATH=/usr/local/lib
ENV CONDA_PREFIX="/root/miniconda3"
ARG DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Kolkata

WORKDIR /tmp

RUN apt-get update && apt-get install git wget -y && rm -rf /var/lib/apt/lists/*

RUN wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && mkdir /root/.conda && bash Miniconda3-latest-Linux-x86_64.sh -b && rm -f Miniconda3-latest-Linux-x86_64.sh


RUN conda create -n mlpack -c conda-forge python pandas mamba cython -y && conda clean --all -y

RUN alias sh=bash && conda init bash &&. ~/.bashrc && echo "conda activate mlpack" >> ~/.bashrc
ENV PATH /root/miniconda3/envs/mlpack/bin:$PATH

RUN wget -q https://raw.githubusercontent.com/mlpack/examples/master/scripts/jupyter-conda-setup.sh

RUN bash jupyter-conda-setup.sh mlpack

WORKDIR /root
