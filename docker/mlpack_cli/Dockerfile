FROM ubuntu:latest

ARG DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Kolkata

RUN apt-get update && apt-get install wget build-essential wget cmake gfortran txt2man pkg-config libbfd-dev -y && rm -rf /var/lib/apt/lists/* && apt-get autoremove

WORKDIR /tmp

RUN wget -q https://github.com/Reference-LAPACK/lapack/archive/refs/tags/v3.10.0.tar.gz
RUN tar xf v3.10.0.tar.gz && cd lapack-3.10.0 && mkdir build && cd build && cmake -DBUILD_SHARED_LIBS=ON .. && cmake --build . -j 4 --target install && cd /tmp && rm -rf lapack-3.10.0 v3.10.0.tar.gz
RUN wget -q https://github.com/xianyi/OpenBLAS/releases/download/v0.3.17/OpenBLAS-0.3.17.tar.gz
RUN tar xf OpenBLAS-0.3.17.tar.gz && cd OpenBLAS-0.3.17 && mkdir build && cd build && cmake -DBUILD_SHARED_LIBS=ON .. && make -j 4 install && cd /tmp && rm -rf OpenBLAS-0.3.17 OpenBLAS-0.3.17.tar.gz
RUN wget -q https://iweb.dl.sourceforge.net/project/arma/armadillo-9.880.1.tar.xz
RUN tar xf armadillo-9.880.1.tar.xz && cd armadillo-9.880.1 && mkdir build && cd build && cmake -DBUILD_SHARED_LIBS=ON .. && make  -j 4 install && cd /tmp && rm -rf armadillo-9.880.1.tar.xz armadillo-9.880.1
RUN wget -q https://boostorg.jfrog.io/artifactory/main/release/1.78.0/source/boost_1_78_0.tar.gz 
RUN tar xvf boost_1_78_0.tar.gz && cd boost_1_78_0 && sh bootstrap.sh && ./b2 install --prefix=/usr/local/ variant=release link=shared threading=multi runtime-link=shared && cd /tmp && rm -rf boost_1_78_0 boost_1_78_0.tar.gz
RUN wget -q http://ensmallen.org/files/ensmallen-2.18.2.tar.gz 
RUN tar xf ensmallen-2.18.2.tar.gz && cd ensmallen-2.18.2 && mkdir build && cd build && cmake -DBUILD_SHARED_LIBS=ON .. && make -j 4 install && cd /tmp && rm -rf ensmallen-2.18.2 ensmallen-2.18.2.tar.gz
RUN wget -q https://github.com/USCiLab/cereal/archive/refs/tags/v1.3.2.tar.gz
RUN tar xf v1.3.2.tar.gz && cd cereal-1.3.2/ && mkdir build && cd build && cmake -DBUILD_SHARED_LIBS=ON -DSKIP_PORTABILITY_TEST=1 .. && make -j 4 install && cd /tmp && rm -rf v1.3.2.tar.gz cereal-1.3.2

RUN wget -q https://github.com/mlpack/mlpack/archive/refs/tags/3.4.2.tar.gz && tar xf 3.4.2.tar.gz && cd mlpack-3.4.2
ENV LD_LIBRARY_PATH=/usr/local/lib

RUN cd /tmp/mlpack-3.4.2 && mkdir build && cd build && cmake -DBUILD_SHARED_LIBS=ON -D USE_OPENMP=ON .. && make -j4 install && cd .. && rm -rf build

WORKDIR /root
