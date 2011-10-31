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
  generate_random_problem.h
  generate_random_problem.cc
  generic_npt_alg.h
  generic_npt_alg_impl.h
  naive_resampling.h
  naive_resampling.cc
  node_tuple.h
  node_tuple.cc
  #n_point_single_main.cc
  #n_point_angle_resampling_main.cpp
  permutations.h
  permutations.cc
  single_driver.h
  single_driver.cc
  single_matcher.h
  single_matcher.cc
  single_results.h
  single_results.cc
  test_angle_resampling.h
  test_angle_resampling.cc
  test_single_bandwidth.h
  test_single_bandwidth.cc
)
# IMPORTANT: do I need the mains here?

# add directory name to sources
set(DIR_SRCS)
foreach(file ${SOURCES})
  set(DIR_SRCS ${DIR_SRCS} ${CMAKE_CURRENT_SOURCE_DIR}/${file})
endforeach()
# append sources (with directory name) to list of all MLPACK sources (used at the parent scope)
set(MLPACK_CONTRIB_SRCS ${MLPACK_CONTRIB_SRCS} ${DIR_SRCS} PARENT_SCOPE)

add_executable(n_point_angle_resampling_main
    EXCLUDE_FROM_ALL
    n_point_angle_resampling_main.cpp
)
target_link_libraries(n_point_angle_resampling_main
    mlpack
    mlpack_contrib
)

add_executable(n_point_single_main
    EXCLUDE_FROM_ALL
    n_point_single_main.cc
)
target_link_libraries(n_point_single_main
    mlpack
    mlpack_contrib
)

add_executable(single_bandwidth_test
	EXCLUDE_FROM_ALL
	single_bandwidth.test.cc
)
target_link_libraries(single_bandwidth_test
#	mlpack
	mlpack_contrib
	boost_unit_test_framework
	boost_random
)

add_executable(angle_resampling_test
	EXCLUDE_FROM_ALL
	angle_resampling.test.cc
)
target_link_libraries(angle_resampling_test
#	mlpack
	mlpack_contrib
	boost_unit_test_framework
	boost_random
)

add_executable(two_point_timing
	EXCLUDE_FROM_ALL
	two_point_timing.cc
)
target_link_libraries(two_point_timing
#	mlpack
	mlpack_contrib
	boost_random
)


