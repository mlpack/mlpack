cmake_minimum_required(VERSION 2.8)

# Define the files we need to compile.
# Anything not in this list will not be compiled into MLPACK.
set(SOURCES
  bounds.h
  dballbound.h
  dballbound_impl.h
  dhrectperiodicbound.h
  dhrectperiodicbound_impl.h
  hrectbound.h
  hrectbound_impl.h
  spacetree.h
  statistic.h
)

# add directory name to sources
set(DIR_SRCS)
foreach(file ${SOURCES})
  set(DIR_SRCS ${DIR_SRCS} ${CMAKE_CURRENT_SOURCE_DIR}/${file})
endforeach()
# Append sources (with directory name) to list of all MLPACK sources (used at
# the parent scope).
set(MLPACK_SRCS ${MLPACK_SRCS} ${DIR_SRCS} PARENT_SCOPE)

# test executable
add_executable(tree_test
  tree_test.cc
)
# link dependencies of test executable
target_link_libraries(tree_test
  mlpack
  boost_unit_test_framework
)
