# Select Image
FROM ubuntu
# Developer Info
MAINTAINER Ryan Curtin <ryan@ratml.org>
# Update Packages
RUN apt-get update && apt-get -y upgrade
# Install Dependencies
RUN apt-get -y install gcc g++ git cmake subversion libboost-math-dev libboost-program-options-dev libboost-test-dev libboost-serialization-dev libarmadillo-dev binutils-dev
# Copy Source
COPY . /mlpack
# Current Directory
WORKDIR /mlpack
# Build mlpack
# Build requires very high spec docker machine with RAM>4GB and CPUs>4
RUN mkdir build && cd build && cmake ../ && make -j8 && make mlpack_test
# Start Shell
CMD ["/bin/bash"]
