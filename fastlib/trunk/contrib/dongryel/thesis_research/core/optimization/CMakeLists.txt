cmake_minimum_required(VERSION 2.6)

set(SOURCES
  lbfgs_dev.h
  lbfgs.h
  quasi_newton_hessian_update.h
  rand_range_finder.h
  trust_region_dev.h
  trust_region.h
)

set(DIR_SRCS)
foreach(file ${SOURCES})
  set(DIR_SRCS ${DIR_SRCS} ${CMAKE_CURRENT_SOURCE_DIR}/${file})
endforeach()

set(CORE_SRCS ${CORE_SRCS} ${DIR_SRCS} PARENT_SCOPE)

# test executable
add_executable(optimization-test
               EXCLUDE_FROM_ALL
               optimization.test.cc
              )
# link dependencies of test executable
target_link_libraries(optimization-test
                      core
                      ${Boost_LIBRARIES}
                     )

# test executable
add_executable(rand_approx-test
               EXCLUDE_FROM_ALL
               rand_approx.test.cc
              )
# link dependencies of test executable
target_link_libraries(rand_approx-test
                      core
                      ${Boost_LIBRARIES}
                     )
