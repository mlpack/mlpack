cmake_minimum_required(VERSION 2.8)

# Define the files we need to compile
# Anything not in this list will not be compiled into the output library
# Do not include test programs here
set(SOURCES
  kde_dual_tree.hpp
  kde_dual_tree_impl.hpp
  test_kde_dual_tree.cpp
)

# add directory name to sources
set(DIR_SRCS)
foreach(file ${SOURCES})
  set(DIR_SRCS ${DIR_SRCS} ${CMAKE_CURRENT_SOURCE_DIR}/${file})
endforeach()
# append sources (with directory name) to list of all MLPACK sources (used at the parent scope)
set(MLPACK_CONTRIB_SRCS ${MLPACK_CONTRIB_SRCS} ${DIR_SRCS} PARENT_SCOPE)

# link dependencies of test executable
add_executable(test_kde_dual_tree
   EXCLUDE_FROM_ALL
   test_kde_dual_tree.cpp
)
target_link_libraries(test_kde_dual_tree
   mlpack
)
