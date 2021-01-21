#!/bin/bash
#
# This script download OpenBLAS dependecy for Armadillo and cross-compile them against
# required architecture.
#
version=1.75.0
version_underscore=1_75_0

gcc_version=$1

echo "Download and build boost as a static lbrary"
	wget --no-check-certificate \
	"https://dl.bintray.com/boostorg/release/$version/source/boost_$version_underscore.tar.gz" && \
	tar xvzf boost_$version_underscore.tar.gz && \
	rm -f boost_$version_underscore.tar.gz && \

echo "Download the Cereal library"
	wget --no-check-certificate \
	"https://github.com/USCiLab/cereal/archive/v1.3.0.tar.gz" && \
	tar xvzf v1.3.0.tar.gz && \
	rm -f v1.3.0.tar.gz && \

echo "Download and build OpenBLAS as a static library"
wget --no-check-certificate \
      "https://github.com/xianyi/OpenBLAS/archive/v0.3.13.tar.gz" && \
       tar xvzf v0.3.13.tar.gz && \
       rm -f v0.3.13.tar.gz && \
       cd OpenBLAS-0.3.13 && \
       make TARGET=ARMV8 BINARY=64 HOSTCC=gcc CC=aarch64-linux-gnu-gcc$gcc_version FC=aarch64-linux-gnu-gfortran$gcc_version NO_SHARED=1 &&\
       make *.a libopenblas.a
       cd ../ &&\

exit 0
