cmake_minimum_required(VERSION 2.8)

# Define the files we need to compile
# Anything not in this list will not be compiled into the output library
# Do not include test programs here
set(SOURCES
   gen_metric_tree.h
   gen_metric_tree_impl.h
   general_spacetree.h
   allknn_balltree.h
)

# add directory name to sources
set(DIR_SRCS)
foreach(file ${SOURCES})
  set(DIR_SRCS ${DIR_SRCS} ${CMAKE_CURRENT_SOURCE_DIR}/${file})
endforeach()
# append sources (with directory name) to list of all MLPACK sources (used at the parent scope)
set(MLPACK_CONTRIB_SRCS ${MLPACK_CONTRIB_SRCS} ${DIR_SRCS} PARENT_SCOPE)

# executable
add_executable(bt_knn
   allknn_balltree_test.cc
)
# link dependencies of executable
target_link_libraries(bt_knn
   fastlib
   mlpack
   mlpack_contrib
)

add_executable(allknn_balltree_graph
   allknn_balltree_construct_graph.cc
)
target_link_libraries(allknn_balltree_graph
   fastlib  
   mlpack
   mlpack_contrib
)
