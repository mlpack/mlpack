cmake_minimum_required(VERSION 2.8)

set(SOURCES
  lbfgs_dev.h
  lbfgs.h
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
               optmization.test.cc
              )
# link dependencies of test executable
target_link_libraries(optimization-test
                      core
                      ${Boost_LIBRARIES}
                     )
