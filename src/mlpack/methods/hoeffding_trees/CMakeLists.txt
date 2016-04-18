# Define the files we need to compile
# Anything not in this list will not be compiled into mlpack.
set(SOURCES
  binary_numeric_split.hpp
  binary_numeric_split_impl.hpp
  binary_numeric_split_info.hpp
  categorical_split_info.hpp
  gini_impurity.hpp
  hoeffding_categorical_split.hpp
  hoeffding_categorical_split_impl.hpp
  hoeffding_numeric_split.hpp
  hoeffding_numeric_split_impl.hpp
  hoeffding_tree.hpp
  hoeffding_tree_impl.hpp
  information_gain.hpp
  numeric_split_info.hpp
  typedef.hpp
)

# Add directory name to sources.
set(DIR_SRCS)
foreach(file ${SOURCES})
  set(DIR_SRCS ${DIR_SRCS} ${CMAKE_CURRENT_SOURCE_DIR}/${file})
endforeach()
# Append sources (with directory name) to list of all mlpack sources (used at
# the parent scope).
set(MLPACK_SRCS ${MLPACK_SRCS} ${DIR_SRCS} PARENT_SCOPE)

add_cli_executable(hoeffding_tree)
