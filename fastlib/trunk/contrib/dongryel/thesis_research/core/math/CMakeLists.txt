cmake_minimum_required(VERSION 2.8)

set(SOURCES
  geometry.cc
  geometry.h
  global.cc
  math_lib.h
  math_lib_impl.h
  range.h
)

set(DIR_SRCS)
foreach(file ${SOURCES})
  set(DIR_SRCS ${DIR_SRCS} ${CMAKE_CURRENT_SOURCE_DIR}/${file})
endforeach()

set(CORE_SRCS ${CORE_SRCS} ${DIR_SRCS} PARENT_SCOPE)

# test executable
add_executable(armadillo_wrapper-test
               EXCLUDE_FROM_ALL
               armadillo_wrapper.test.cc
              )
# link dependencies of test executable
target_link_libraries(armadillo_wrapper-test
                      core
                      ${Boost_LIBRARIES}
                     )
