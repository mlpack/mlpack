cmake_minimum_required(VERSION 2.8)

set(SOURCES
  lbfgs_impl.hpp
  lbfgs.hpp
  test_functions.hpp
  test_functions.cpp
)

set(DIR_SRCS)
foreach(file ${SOURCES})
  set(DIR_SRCS ${DIR_SRCS} ${CMAKE_CURRENT_SOURCE_DIR}/${file})
endforeach()

set(MLPACK_SRCS ${MLPACK_SRCS} ${DIR_SRCS} PARENT_SCOPE)

# test executable
add_executable(lbfgs_test
  EXCLUDE_FROM_ALL
  lbfgs_test.cpp
)
# link dependencies of test executable
target_link_libraries(lbfgs_test
  mlpack
  boost_unit_test_framework
)
