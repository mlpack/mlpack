cmake_minimum_required(VERSION 2.8)

# We have subdirectories in this project
add_subdirectory(epxtl_gc)
#add_subdirectory(exptl)  ## appears deprecated?

# Define the files we need to compile
# Anything not in this list will not be compiled into the output library
# Do not include test programs here
set(SOURCES
   ctree.h
   ctree_impl.h
   cover_tree.h
   allknn.h
   allknn_impl.h
   distances.h
)

# add directory name to sources
set(DIR_SRCS)
foreach(file ${SOURCES})
  set(DIR_SRCS ${DIR_SRCS} ${CMAKE_CURRENT_SOURCE_DIR}/${file})
endforeach()
# append sources (with directory name) to list of all MLPACK sources (used at the parent scope)
set(MLPACK_CONTRIB_SRCS ${MLPACK_CONTRIB_SRCS} ${DIR_SRCS} PARENT_SCOPE)

# executable
add_executable(cover_tree_test
   cover_tree_test.cc
)
# link dependencies of executable
target_link_libraries(cover_tree_test
   fastlib
   mlpack
   mlpack_contrib
)
