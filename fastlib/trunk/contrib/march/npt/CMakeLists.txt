cmake_minimum_required(VERSION 2.8)

# Define the files we need to compile
# Anything not in this list will not be compiled into the output library
# Do not include test programs here
set(SOURCES
   matcher.h
   matcher.cc
   n_point.h
   n_point.cc
   n_point_impl.h
   n_point_impl.cc
   n_point_nodes.h
   n_point_nodes.cc
   n_point_perm_free.h
   n_point_perm_free.cc
   perm_free_matcher.h
   perm_free_matcher.cc
)

# add directory name to sources
set(DIR_SRCS)
foreach(file ${SOURCES})
  set(DIR_SRCS ${DIR_SRCS} ${CMAKE_CURRENT_SOURCE_DIR}/${file})
endforeach()
# append sources (with directory name) to list of all MLPACK sources (used at the parent scope)
set(MLPACK_CONTRIB_SRCS ${MLPACK_CONTRIB_SRCS} ${DIR_SRCS} PARENT_SCOPE)

# executable
add_executable(n_point_testing
   EXCLUDE_FROM_ALL
   n_point_testing.h
   n_point_testing.cc
)
# link dependencies of executable
target_link_libraries(n_point_testing
   fastlib
   mlpack
   mlpack_contrib
)

add_executable(n_point
   EXCLUDE_FROM_ALL
   n_point_main.cc
)
target_link_libraries(n_point
   fastlib
   mlpack
   mlpack_contrib
)
