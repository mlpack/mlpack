#!/bin/bash
#
# This script download OpenBLAS dependecy for Armadillo and cross-compile them against
# required architecture.
# 
echo "Download and build OpenBLAS as a static library"
wget --no-check-certificate \
      "https://github.com/xianyi/OpenBLAS/archive/v0.3.9.tar.gz" && \
       tar xvzf v0.3.9.tar.gz && \
       rm -f v0.3.9.tar.gz && \
       cd OpenBLAS-0.3.9 && \
       make TARGET=ARMV8 BINARY=64 HOSTCC=gcc CC=aarch64-linux-gnu-gcc NOFORTRAN=1 libs &&\
       cd ../ &&\
exit 0
