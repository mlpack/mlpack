cmake_minimum_required(VERSION 2.8)
project(FASTLIB C CXX Fortran)

## External Libraries
#  ls /usr/share/cmake-2.6/Modules/Find* | \
#  perl -ne 's#.*Modules/Find(.*)>cmake#\1#; print'

# set path right
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/CMake")

# find LAPACK
find_package(LAPACK REQUIRED)  # LAPACK finds BLAS as a dependency

# find pthreads
find_package(Pthreads REQUIRED)

# include boost.
set(Boost_USE_STATIC_LIBS   ON)
find_package(Boost COMPONENTS program_options mpi serialization system thread unit_test_framework REQUIRED)
include_directories(${Boost_INCLUDE_DIRS})

# include trilinos.
set(TRILINOS_REQUIRED_LIBS epetra anasazi aztecoo ifpack teuchos ml triutils epetraext belos amesos)
find_package(Trilinos REQUIRED)

# Set the MPI compiler.
set(MPI_COMPILER mpicxx.openmpi)

# find MPI
find_package(MPI)

if(MPI_FOUND)
  list(APPEND TRILINOS_LIBS ${MPI_LIBRARIES})
  include_directories(${MPI_INCLUDE_PATH})
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${MPI_COMPILE_FLAGS}")
  set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${MPI_LINK_FLAGS}")
endif()

# find CUDA
find_package(CUDA)

# Add debug flags.
set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} -Wall")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -Wall")

# Add the Electric fence.
set(CMAKE_EXE_LINKER_FLAGS_DEBUG "-lefence")

include_directories(${TRILINOS_INCLUDE_DIR})
include_directories(${MPI_INCLUDE_DIR})

# for a local installation of armadillo
include_directories($ENV{HOME}/local/armadillo-1.0.0/include)
link_directories($ENV{HOME}/local/armadillo-1.0.0/lib)

# distclean option because cmake doesn't support it
include(CMake/TargetDistclean.cmake OPTIONAL)

#include_directories( ${FASTLIB_INCLUDE_DIRS} )
include_directories(${CMAKE_BINARY_DIR}/include/)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/)

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib/)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib/)

## recurse
add_subdirectory(core)
add_subdirectory(mlpack)
add_subdirectory(physpack)
