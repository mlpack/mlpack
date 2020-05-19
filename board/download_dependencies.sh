#!/bin/bash
#
# This script download all dependecies for mlpack and cross-compile them against
# required architecture.
# 

version=$1
version_underscore=$2


echo "Let us download boost as dependecies..."
wget --no-check-certificate \
      "https://dl.bintray.com/boostorg/release/$version/source/boost_$version_underscore.tar.gz" && \
    tar xvzf boost_$version_underscore.tar.gz && \
    rm -f boost_$version_underscore.tar.gz && \
    cd boost_$version_underscore && \
    cp tools/build/example/user-config.jam project-config.jam && \
    echo 'using gcc : arm : arm-linux-gnueabihf-g++ ;' >> project-config.jam &&\
    b2 --build=$PWD/build toolset=gcc-arm link=static cxxflags=-fPIC && \
    -with-libraries=math,program_options,serialization,test && \
    cd ../ && \
    rm -rf boost_$version_underscore
