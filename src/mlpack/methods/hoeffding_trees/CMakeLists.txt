# Define the files we need to compile
# Anything not in this list will not be compiled into MLPACK.
set(SOURCES
  categorical_split_info.hpp
  gini_impurity.hpp
  hoeffding_categorical_split.hpp
  hoeffding_categorical_split_impl.hpp
  hoeffding_numeric_split.hpp
  hoeffding_numeric_split_impl.hpp
  hoeffding_split.hpp
  hoeffding_split_impl.hpp
  numeric_split_info.hpp
  streaming_decision_tree.hpp
  streaming_decision_tree_impl.hpp
)

# Add directory name to sources.
set(DIR_SRCS)
foreach(file ${SOURCES})
  set(DIR_SRCS ${DIR_SRCS} ${CMAKE_CURRENT_SOURCE_DIR}/${file})
endforeach()
# Append sources (with directory name) to list of all MLPACK sources (used at
# the parent scope).
set(MLPACK_SRCS ${MLPACK_SRCS} ${DIR_SRCS} PARENT_SCOPE)

add_executable(streaming_decision_tree
  streaming_decision_tree_main.cpp
)
target_link_libraries(streaming_decision_tree
  mlpack
)
