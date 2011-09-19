cmake_minimum_required(VERSION 2.6)
project(FASTLIB C CXX Fortran)

add_subdirectory(cuda_kde)
add_subdirectory(distributed_kde)
add_subdirectory(distributed_kpca)
add_subdirectory(distributed_local_regression)
add_subdirectory(gp_regression)
add_subdirectory(kde)
add_subdirectory(local_regression)
add_subdirectory(mixed_logit_dcm)
add_subdirectory(series_expansion)

#added by Bill, 9/12
add_subdirectory(distributed_2pt)
add_subdirectory(two_point)
