# Select Image
FROM ubuntu
# Developer Info
MAINTAINER Ryan Curtin <ryan@ratml.org>
# Update Packages
RUN apt-get update && apt-get -y upgrade
# Install Dependencies
RUN apt-get -y install gcc g++ git cmake subversion \
    libboost-math-dev libboost-program-options-dev \
    libboost-test-dev libboost-serialization-dev \
    libarmadillo-dev binutils-dev
# Copy Source
COPY . /mlpack
# Build requires very high spec docker machine with RAM>4GB and CPUs>2 if we use multiple jobs
# Using lesser jobs slows down the compilation process but works perfectly
RUN cd /mlpack && mkdir build && cd build && cmake -D DEBUG=OFF -D PROFILE=OFF ../ && make && make install
# Setup Development Environment
WORKDIR /mlpack/build
# Setup Environment variable
ENV LD_LIBRARY_PATH /usr/local/lib
# Start Shell
CMD ["/bin/bash"]
