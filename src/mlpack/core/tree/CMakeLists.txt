cmake_minimum_required(VERSION 2.8)

# Define the files we need to compile.
# Anything not in this list will not be compiled into MLPACK.
set(SOURCES
  ballbound.hpp
  ballbound_impl.hpp
  binary_space_tree.hpp
  binary_space_tree_impl.hpp
  bounds.hpp
  cover_tree.hpp
  cover_tree_impl.hpp
  first_point_is_root.hpp
  hrectbound.hpp
  hrectbound_impl.hpp
  periodichrectbound.hpp
  periodichrectbound_impl.hpp
  statistic.hpp
  mrkd_statistic.hpp
  traversers/single_tree_breadth_first_traverser.hpp
  traversers/single_tree_depth_first_traverser.hpp
  traversers/dual_tree_depth_first_traverser.hpp
)

# add directory name to sources
set(DIR_SRCS)
foreach(file ${SOURCES})
  set(DIR_SRCS ${DIR_SRCS} ${CMAKE_CURRENT_SOURCE_DIR}/${file})
endforeach()
# Append sources (with directory name) to list of all MLPACK sources (used at
# the parent scope).
set(MLPACK_SRCS ${MLPACK_SRCS} ${DIR_SRCS} PARENT_SCOPE)
