cmake_minimum_required(VERSION 2.8)

set(SOURCES
  aug_lagrangian.h
  aug_lagrangian_impl.h
  aug_lagrangian_test_functions.h
  aug_lagrangian_test_functions.cc
)

set(DIR_SRCS)
foreach(file ${SOURCES})
  set(DIR_SRCS ${DIR_SRCS} ${CMAKE_CURRENT_SOURCE_DIR}/${file})
endforeach()

set(MLPACK_SRCS ${MLPACK_SRCS} ${DIR_SRCS} PARENT_SCOPE)

# test executable
add_executable(aug_lagrangian_test
  EXCLUDE_FROM_ALL
  aug_lagrangian_test.cc
)
target_link_libraries(aug_lagrangian_test
  mlpack
  boost_unit_test_framework
)
