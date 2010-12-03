cmake_minimum_required(VERSION 2.8)
project(TESTMPI C CXX Fortran)
 
set(CMAKE_PREFIX_PATH /opt/boost)
set(CMAKE_CXX_COMPILER mpiCC)
find_package(Boost COMPONENTS mpi REQUIRED)
find_package(LAPACK REQUIRED)
 
link_directories(/opt/armadillo/lib64)
include_directories(/opt/armadillo/include)

add_definitions( -Wall -O2 )

add_executable(test test.cc)
target_link_libraries(test 
  armadillo
  lapack
  blas
  ${Boost_MPI_LIBRARY}
)

add_executable(arma_test arma_test.cc)
target_link_libraries(arma_test
  armadillo
  ${BLAS_blas_LIBRARY}
  ${LAPACK_lapack_LIBRARY}
)
