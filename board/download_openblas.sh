#!/bin/bash
#
# This script download OpenBLAS dependecy for Armadillo and cross-compile them against
# required architecture.
#
version=1.75.0
version_underscore=1_75_0

echo "Download and build boost as a static lbrary"
wget --no-check-certificate \
      "https://dl.bintray.com/boostorg/release/$version/source/boost_$version_underscore.tar.gz" && \
      tar xvzf boost_$version_underscore.tar.gz && \
      rm -f boost_$version_underscore.tar.gz && \

echo "Download and build OpenBLAS as a static library"
wget --no-check-certificate \
      "https://github.com/xianyi/OpenBLAS/archive/v0.3.9.tar.gz" && \
       tar xvzf v0.3.9.tar.gz && \
       rm -f v0.3.9.tar.gz && \
       cd OpenBLAS-0.3.9 && \
       make TARGET=ARMV8 BINARY=64 HOSTCC=gcc CC=aarch64-linux-gnu-gcc NOFORTRAN=1 libs &&\
       cd ../ &&\

echo "Download the Cereal library"
wget --no-check-certificate \
      "https://github.com/USCiLab/cereal/archive/v1.3.0.tar.gz" && \
       tar xvzf v1.3.0.tar.gz && \
       rm -f v1.3.0.tar.gz && \

exit 0
