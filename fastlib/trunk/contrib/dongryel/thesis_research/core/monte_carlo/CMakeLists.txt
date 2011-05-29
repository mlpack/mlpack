cmake_minimum_required(VERSION 2.6)
project(FASTLIB C CXX Fortran)

set(SOURCES
  mean_variance_pair.h
  mean_variance_pair_matrix.h
  multitree_monte_carlo_dev.h
  multitree_monte_carlo.h
  strata.h
  stratum.h
)

set(DIR_SRCS)
foreach(file ${SOURCES})
  set(DIR_SRCS ${DIR_SRCS} ${CMAKE_CURRENT_SOURCE_DIR}/${file})
endforeach()

set(CORE_SRCS ${CORE_SRCS} ${DIR_SRCS} PARENT_SCOPE)

# test executable
add_executable(monte_carlo-test
               EXCLUDE_FROM_ALL
               monte_carlo.test.cc
              )
# link dependencies of test executable
target_link_libraries(monte_carlo-test
                      core
                      ${Boost_LIBRARIES}
                     )
