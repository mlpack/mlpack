# Select Image
FROM gcc:5
# Developer Info
MAINTAINER Ryan Curtin <ryan@ratml.org>
# Update Packages
RUN apt-get update && apt-get -y upgrade
# Install Dependencies
RUN apt-get -y install git cmake subversion libboost-math-dev libboost-program-options-dev libboost-test-dev libboost-serialization-dev libarmadillo-dev binutils-dev
# Copy Source
COPY . /mlpack
# Current Directory
WORKDIR /mlpack
# Build mlpack
RUN mkdir build && cd build && cmake ../ && make && make mlpack_test
# Start Shell
CMD ["/bin/bash"]
