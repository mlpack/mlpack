cmake_minimum_required(VERSION 2.8)
project(FASTLIB C CXX Fortran)

## External Libraries
#  ls /usr/share/cmake-2.6/Modules/Find* | \
#  perl -ne 's#.*Modules/Find(.*)>cmake#\1#; print'

# set path right
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/CMake")

find_package(BLAS REQUIRED)
find_package(LAPACK REQUIRED)
find_package(Pthreads REQUIRED)
find_package(Boost 1.33 COMPONENTS serialization program_options REQUIRED)

#include(CMake/FastlibMacros.cmake)

# distclean option because cmake doesn't support it
include( CMake/TargetDistclean.cmake OPTIONAL )

## Global compiler flags (dep. on platform TODO)
add_definitions( -DDISABLE_DISK_MATRIX )

#include_directories( ${FASTLIB_INCLUDE_DIRS} )
include_directories(${CMAKE_BINARY_DIR}/include/)
#TODO link_directories()

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib/)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib/)

## recurse
add_subdirectory(src) # fastlib
add_subdirectory(mlpack) # mlpack
add_subdirectory(contrib) # mlpack_contrib
#add_subdirectory(physpack) # physpack  ## do not build by default
