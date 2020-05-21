#!/bin/bash
#
# This script download all dependecies for mlpack and cross-compile them against
# required architecture.
# 

version=$1
version_underscore=$2
arch=$3
binary=$4

# echo "Download and build boost as a static lbrary"
# wget --no-check-certificate \
#       "https://dl.bintray.com/boostorg/release/$version/source/boost_$version_underscore.tar.gz" && \
#     tar xvzf boost_$version_underscore.tar.gz && \
#     rm -f boost_$version_underscore.tar.gz && \
#     cd boost_$version_underscore && \
#     cp tools/build/example/user-config.jam project-config.jam && \
#     echo 'using gcc : arm : aarch64-linux-gnu-g++ ;' >> project-config.jam &&\
#     b2 --build=$PWD/build toolset=gcc-arm link=static cxxflags=-fPIC && \
#     -with-libraries=math,program_options,serialization,test && \
#     cd ../../ && \


# echo "Downlaod and build OpenBLAS as a static library"
# wget --no-check-certificate \
#       "https://github.com/xianyi/OpenBLAS/archive/v0.3.9.tar.gz" && \
#       tar xvzf v0.3.9.tar.gz && \
#       rm -f v0.3.9.tar.gz && \
#       cd OpenBLAS-0.3.9 && \
#       make TARGET=ARMV8 BINARY=64 HOSTCC=gcc CC=aarch64-linux-gnu-gcc NOFORTRAN=1 libs &&\
#       cd ../ &&\

# echo "Download and build armadillo as a header"
# wget --no-check-certificate \
#     http://sourceforge.net/projects/arma/files/armadillo-9.880.1.tar.xz && \
#     tar xvf armadillo-9.880.1.tar.xz && \
#     rm -f armadillo-9.880.1.tar.xz && \
#     cd  armadillo-9.880.1

echo "Download and build armadillo as a header"
wget --no-check-certificate \
    https://github.com/Reference-LAPACK/lapack/archive/v3.9.0.tar.gz && \
    tar xvf v3.9.0.tar.gz && \
    rm -f v3.9.0.tar.xz && \
    cp fortran/make.inc lapack-3.9.0
    cd lapack-3.9.0
    make lapacklib -j32

