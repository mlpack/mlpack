cmake_minimum_required(VERSION 2.8)

# Define the files we need to compile.
# Anything not in this list will not be compiled into MLPACK.
set(SOURCES
  lmetric.h
  lmetric.cc
  mahalanobis_distance.h
  mahalanobis_distance_impl.h
  cosine_distance.h
  cosine_distance.cc
)

# add directory name to sources
set(DIR_SRCS)
foreach(file ${SOURCES})
  set(DIR_SRCS ${DIR_SRCS} ${CMAKE_CURRENT_SOURCE_DIR}/${file})
endforeach()
# Append sources (with directory name) to list of all MLPACK sources (used at
# the parent scope).
set(MLPACK_SRCS ${MLPACK_SRCS} ${DIR_SRCS} PARENT_SCOPE)

# add test executable
add_executable(kernel_test
  kernel_test.cc
)
target_link_libraries(kernel_test
  fastlib
  mlpack
  boost_unit_test_framework
)
