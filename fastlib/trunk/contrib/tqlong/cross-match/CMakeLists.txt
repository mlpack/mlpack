cmake_minimum_required(VERSION 2.8)
project(CROSSMATCH C CXX Fortran)

find_package(Boost COMPONENTS program_options REQUIRED)
#find_package(LAPACK REQUIRED)
#link_directories(/opt/armadillo/lib64)
#include_directories(/opt/armadillo/include)

# compiler flags
add_definitions( -Wall -O2 )

add_executable(cross-match main.cpp)
target_link_libraries(cross-match
  #armadillo
  #lapack
  #blas
  #${Boost_MPI_LIBRARY}
  ${Boost_PROGRAM_OPTIONS_LIBRARY}
)

