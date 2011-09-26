cmake_minimum_required(VERSION 2.6)

set(SOURCES
  two_point_arguments.h
  two_point_delta.h
  two_point_dev.h
  two_point_global.h
  two_point_postponed.h
  two_point_result.h
  two_point_statistic.h
  two_point.cc
  two_point.h
  two_point_dualtree.h
)
# IMPORTANT: do I need the mains here?

# add directory name to sources
set(DIR_SRCS)
foreach(file ${SOURCES})
  set(DIR_SRCS ${DIR_SRCS} ${CMAKE_CURRENT_SOURCE_DIR}/${file})
endforeach()
# append sources (with directory name) to list of all MLPACK sources (used at the parent scope)
set(MLPACK_CONTRIB_SRCS ${MLPACK_CONTRIB_SRCS} ${DIR_SRCS} PARENT_SCOPE)



# the main driver for two_point
add_executable(two_point
               EXCLUDE_FROM_ALL
               two_point.cc
              )
# link dependencies of two_point
target_link_libraries(two_point
                      core
                      ${Boost_LIBRARIES}
                     )

# the test driver 
add_executable(two_point_test
               EXCLUDE_FROM_ALL
               two_point.test.cc
              )
# link dependencies t
target_link_libraries(two_point_test
                      core
                      ${Boost_LIBRARIES}
                     )
