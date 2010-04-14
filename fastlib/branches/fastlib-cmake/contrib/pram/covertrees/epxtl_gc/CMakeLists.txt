cmake_minimum_required(VERSION 2.8)

# Define the files we need to compile
# Anything not in this list will not be compiled into the output library
# Do not include test programs here
set(SOURCES
   ctree_gc.h
   ctree_impl.h
   cover_tree_gc.h
   allknn_gc.h
   allknn_impl.h
   distances_gc.h
   gonzalez_gc.h
)

# add directory name to sources
set(DIR_SRCS)
foreach(file ${SOURCES})
  set(DIR_SRCS ${DIR_SRCS} ${CMAKE_CURRENT_SOURCE_DIR}/${file})
endforeach()
# append sources (with directory name) to list of all MLPACK sources (used at the parent scope)
set(MLPACK_CONTRIB_SRCS ${MLPACK_CONTRIB_SRCS} ${DIR_SRCS} PARENT_SCOPE)

# executable
add_executable(cover_tree_test_gc
   cover_tree_test_gc.cc
)
# link dependencies of executable
target_link_libraries(cover_tree_test_gc
   fastlib
   mlpack
   mlpack_contrib
)
