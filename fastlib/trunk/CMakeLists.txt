cmake_minimum_required(VERSION 2.8)
project(FASTLIB C CXX Fortran)

## External Libraries
#  ls /usr/share/cmake-2.6/Modules/Find* | \
#  perl -ne 's#.*Modules/Find(.*)>cmake#\1#; print'

# set path right
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/CMake")

find_package(LAPACK REQUIRED)  # LAPACK finds BLAS as a dependency
find_package(Pthreads REQUIRED)

option(WITH_OPTIMIZERS "Include optimization code - requires OPT++." OFF)
option(WITH_SPARSE "Include sparse matrix code - requires Trilinos." ON)
option(SCALE_NORMAL "Compile with supported indices up to native int type" OFF)
option(SCALE_LARGE "Support indices up to the maximum the machine's architecture can support" OFF)
option(SCALE_MASSIVE "Use 64-bit indexes regardless of architecture" OFF)

if(WITH_SPARSE)
   set(TRILINOS_REQUIRED_LIBS epetra anasazi aztecoo ifpack teuchos)
   find_package(Trilinos REQUIRED)

   include_directories(${TRILINOS_INCLUDE_DIR})
   include_directories(${MPI_INCLUDE_DIR})
endif(WITH_SPARSE)

# ensure we are only using one of the SCALE_x options
if(SCALE_MASSIVE)
   if(SCALE_LARGE OR SCALE_NORMAL)
      message(FATAL_ERROR "Only one of SCALE_MASSIVE, SCALE_LARGE, or SCALE_NORMAL options allowed")
   endif()

   add_definitions(-DSCALE_MASSIVE)
elseif(SCALE_LARGE)
   if(SCALE_NORMAL)
      message(FATAL_ERROR "Only one of SCALE_MASSIVE, SCALE_LARGE, or SCALE_NORMAL options allowed")
   endif()

   add_definitions(-DSCALE_LARGE)
else()
   # default to normal scale
   add_definitions(-DSCALE_NORMAL)
endif()

# distclean option because cmake doesn't support it
include(CMake/TargetDistclean.cmake OPTIONAL)

## Global compiler flags (dep. on platform TODO)
add_definitions(-DDISABLE_DISK_MATRIX)

#include_directories( ${FASTLIB_INCLUDE_DIRS} )
include_directories(${CMAKE_BINARY_DIR}/include/)
#TODO link_directories()

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib/)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib/)

## recurse
add_subdirectory(fastlib) # fastlib
add_subdirectory(mlpack) # mlpack
add_subdirectory(contrib) # mlpack_contrib
#add_subdirectory(physpack) # physpack  ## do not build by default

## install all binaries
install(DIRECTORY ${CMAKE_BINARY_DIR}/bin/ 
    DESTINATION bin/
    FILE_PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_READ GROUP_EXECUTE WORLD_READ WORLD_EXECUTE
    FILES_MATCHING PATTERN * PATTERN *_test EXCLUDE)
