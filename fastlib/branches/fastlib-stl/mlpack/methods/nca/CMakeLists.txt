cmake_minimum_required(VERSION 2.8)

# Define the files we need to compile
# Anything not in this list will not be compiled into the output library
# Do not include test programs here
set(SOURCES
  nca.h
  nca_impl.h
  nca_softmax_error_function.h
  nca_softmax_error_function_impl.h
)

# add directory name to sources
set(DIR_SRCS)
foreach(file ${SOURCES})
  set(DIR_SRCS ${DIR_SRCS} ${CMAKE_CURRENT_SOURCE_DIR}/${file})
endforeach()
# append sources (with directory name) to list of all MLPACK sources (used at
# the parent scope)
set(MLPACK_SRCS ${MLPACK_SRCS} ${DIR_SRCS} PARENT_SCOPE)

add_executable(nca
  nca_main.cc
)
target_link_libraries(nca
  fastlib
  mlpack
)

# test executable
add_executable(nca_test
  nca_test.cc
)
target_link_libraries(nca_test
  fastlib
  mlpack
  boost_unit_test_framework
)
