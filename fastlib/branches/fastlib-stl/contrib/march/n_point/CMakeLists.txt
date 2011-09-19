cmake_minimum_required(VERSION 2.8)

# Define the files we need to compile
# Anything not in this list will not be compiled into the output library
# Do not include test programs here
set(SOURCES
  angle_driver.h
  angle_driver.cc
  angle_matcher_generation.h
  angle_matcher_generation.cc
  angle_matcher.h
  angle_matcher.cc
  angle_results.h
  angle_results.cc  
  efficient_resampling.h
  efficient_resampling.cc
  generic_npt_alg.h
  generic_npt_alg_impl.h
  naive_resampling.h
  naive_resampling.cc
  node_tuple.h
  node_tuple.cc
  permutations.h
  permutations.cc
  single_driver.h
  single_driver.cc
  single_matcher.h
  single_matcher.cc
  single_results.h
  single_results.cc
)
# IMPORTANT: do I need the mains here?

# add directory name to sources
set(DIR_SRCS)
foreach(file ${SOURCES})
  set(DIR_SRCS ${DIR_SRCS} ${CMAKE_CURRENT_SOURCE_DIR}/${file})
endforeach()
# append sources (with directory name) to list of all MLPACK sources (used at the parent scope)
set(MLPACK_CONTRIB_SRCS ${MLPACK_CONTRIB_SRCS} ${DIR_SRCS} PARENT_SCOPE)

# executable
#add_executable(n_point_test
#   EXCLUDE_FROM_ALL
#   n_point_testing.h
#   n_point_testing.cc
#)
# link dependencies of executable
#target_link_libraries(n_point_testing
#   fastlib
#   mlpack
#   mlpack_contrib
#)

add_executable(n_point_angle_resampling_main
    EXCLUDE_FROM_ALL
    n_point_angle_resampling_main.cpp
)
target_link_libraries(n_point_angle_resampling_main
    fastlib
    mlpack
    mlpack_contrib
)

add_executable(n_point_single_main
    EXCLUDE_FROM_ALL
    n_point_single_main.cc
)
target_link_libraries(n_point_single_main
    fastlib
    mlpack
    mlpack_contrib
)

