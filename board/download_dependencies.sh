#!/bin/bash
#
# This script download OpenBLAS dependecy for Armadillo and cross-compile them against
# required architecture.
#
board_name=$1

echo "Download and build boost as a static lbrary"
wget --no-check-certificate \
    "https://dl.bintray.com/boostorg/release/1.75.0/source/boost_1_75_0.tar.gz" && \
    tar xvzf boost_1_75_0.tar.gz && \
    rm -f boost_1.75.0.tar.gz && \

echo "Download the Cereal library"
wget --no-check-certificate \
    "https://github.com/USCiLab/cereal/archive/v1.3.0.tar.gz" && \
    tar xvzf v1.3.0.tar.gz && \
    rm -f v1.3.0.tar.gz && \

echo "Download the Armadillo library"
    curl https://data.kurg.org/armadillo-8.400.0.tar.xz | tar -xvJ

echo "Download and build OpenBLAS as a static library"
wget --no-check-certificate \
      "https://github.com/xianyi/OpenBLAS/archive/v0.3.13.tar.gz" && \
       tar xvzf v0.3.13.tar.gz && \
       rm -f v0.3.13.tar.gz && \
       cd OpenBLAS-0.3.13 && \
       if [[ "$board_name" == "rpi3" ]]
       then
         make TARGET=ARMV8 BINARY=64 HOSTCC=gcc CC=aarch64-linux-gnu-gcc FC=aarch64-linux-gnu-gfortran NO_SHARED=1
       fi
       cd ../ &&\

exit 0
