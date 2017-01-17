cmake_minimum_required(VERSION 2.8)

# Define the files we need to compile.
# Anything not in this list will not be compiled into mlpack.
set(SOURCES
  decision_tree.hpp
  decision_tree_impl.hpp
  all_categorical_split.hpp
  all_categorical_split_impl.hpp
  best_binary_numeric_split.hpp
  best_binary_numeric_split_impl.hpp
  gini_gain.hpp
  information_gain.hpp
)

# Add directory name to sources.
set(DIR_SRCS)
foreach(file ${SOURCES})
  set(DIR_SRCS ${DIR_SRCS} ${CMAKE_CURRENT_SOURCE_DIR}/${file})
endforeach()
# Append sources (with directory name) to list of all mlpack sources (used at
# the parent scope).
set(MLPACK_SRCS ${MLPACK_SRCS} ${DIR_SRCS} PARENT_SCOPE)
