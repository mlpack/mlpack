cmake_minimum_required(VERSION 2.8)

# Define the files we need to compile
# Anything not in this list will not be compiled into the output library
# Do not include test programs here
set(SOURCES
   bounds.h
   lmetric.h
   dhrectbound.h
   dhrectbound_impl.h
   dballbound.h
   dballbound_impl.h
   kdtree.h
   kdtree_impl.h
   kdtree_impl.cc
   spacetree.h
   statistic.h
)

# add directory name to sources
set(DIR_SRCS)
foreach(file ${SOURCES})
  set(DIR_SRCS ${DIR_SRCS} ${CMAKE_CURRENT_SOURCE_DIR}/${file})
endforeach()
# append sources (with directory name) to list of all FASTLIB sources (used at the parent scope)
set(FASTLIB_SRCS ${FASTLIB_SRCS} ${DIR_SRCS} PARENT_SCOPE)

#add_library(tree ${SOURCES})
# link dependencies of tree
#target_link_libraries(tree
#   base
#   col
#   data
#   file
#   fx
#   la
#)

# test executable
add_executable(tree_test
   tree_test.cc
)
# link dependencies of test executable
target_link_libraries(tree_test
#   tree
   fastlib
)
