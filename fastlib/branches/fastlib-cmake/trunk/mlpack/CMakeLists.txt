cmake_minimum_required(VERSION 2.6)
project(MLPACK C CXX Fortran)

find_package(BLAS REQUIRED)
find_package(LAPACK REQUIRED)

## Global compiler flags (dep. on platform TODO)
set(CMAKE_CXX_FLAGS "-g -Wall")
add_definitions( -DDISABLE_DISK_MATRIX )

## Location of fastlib headers & lib
set(FASTLIB_BASE_DIR /usr/local CACHE PATH
  "Directory where fastlib is installed." )
set(FASTLIB_LIB_DIR ${FASTLIB_BASE_DIR}/lib )
set(FASTLIB_INCLUDE_DIR ${FASTLIB_BASE_DIR}/include )
# set(FASTLIB_LIB_DIR ${FASTLIB_BASE_DIR}/lib CACHE PATH
#   "Directory where the fastlib library is installed." )
# set(FASTLIB_INCLUDE_DIR ${FASTLIB_BASE_DIR}/include CACHE PATH
#   "Directory where the fastlib headers are installed." )
#     
# mark_as_advanced(FASTLIB_LIB_DIR FASTLIB_INCLUDE_DIR) 

##? add_library ?
find_library(FASTLIB_LIB fastlib ${FASTLIB_LIB_DIR})

## include dirs
include_directories( ${FASTLIB_INCLUDE_DIR} )
include_directories( .. )  # mlpack

## link directories
link_directories( ${FASTLIB_LIB_DIR} )

## standard libs
set(MLPACK_STD_LIBS fastlib lapack blas )  ##! this ordering matters :(

## recurse
set( DIRS
    allkfn
    allknn allnn
    # disk_allnn  #requires mmanager
    fastica
    hmm
    infomax_ica
    kalman
    kde
    # kernel_pca  # requires sparse 
    mog
    mvu
    naive_bayes
    quicsvd
    # range_search  ##! uses contrib code
    regression
    series_expansion
    svm
)

foreach(dir ${DIRS})
    add_subdirectory(${dir})
endforeach()

## installation
