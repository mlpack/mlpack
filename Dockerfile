FROM ubuntu

RUN apt-get update -qq && \
    apt-get install -y --no-install-recommends python-pip cmake wget sudo \
    libopenblas-dev liblapack-dev binutils-dev libboost-all-dev pkg-config \
    make txt2man git doxygen libarmadillo-dev build-essential

# Get mlpack version.
ARG mlpack_archive

# Update software repository, install dependencies and build mlpack.
RUN useradd -m mlpack && echo "mlpack:mlpack" | chpasswd && \
    adduser mlpack sudo && su mlpack && cd /home/mlpack/

WORKDIR /home/mlpack/
# RUN wget -O mlpack.tar.gz ${mlpack_archive} && \
COPY ./ /home/mlpack/
RUN  mkdir build
WORKDIR /home/mlpack/build
RUN cmake .. && make && make install && chown -R mlpack:mlpack /home/mlpack

ENV PROJ_WORK_DIR /home/mlpack/build/

# Setup environment.
ENV LD_LIBRARY_PATH /usr/local/lib/
USER mlpack
WORKDIR /home/mlpack
CMD /bin/bash
