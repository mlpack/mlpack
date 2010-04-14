cmake_minimum_required(VERSION 2.8)

# Define the files we need to compile
# Anything not in this list will not be compiled into the output library
# Do not include test programs here
set(SOURCES
   subspace_stat.h
   cfmm_tree.h
   cfmm_tree_impl.h
   general_type_bounds.h
   gen_hypercube_tree.h
   gen_hypercube_tree_impl.h
   gen_hypercube_tree_util.h
   gen_range.h
   gen_range.cc
   gen_kdtree.h
   gen_kdtree_impl.h
   gen_kdtree_hyper.h
   gen_kdtree_hyper_impl.h
   gen_metric_tree.h
   gen_metric_tree_impl.h
   general_spacetree.h
)

# add directory name to sources
set(DIR_SRCS)
foreach(file ${SOURCES})
  set(DIR_SRCS ${DIR_SRCS} ${CMAKE_CURRENT_SOURCE_DIR}/${file})
endforeach()
# append sources (with directory name) to list of all MLPACK sources (used at the parent scope)
set(MLPACK_CONTRIB_SRCS ${MLPACK_CONTRIB_SRCS} ${DIR_SRCS} PARENT_SCOPE)

# executable
add_executable(cfmm_tree_test
   cfmm_tree_test.cc
)
# link dependencies of executable
target_link_libraries(cfmm_tree_test
   fastlib
   mlpack
   mlpack_contrib
)

add_executable(gen_hypercube_tree_test
   gen_hypercube_tree_test.cc
)
target_link_libraries(gen_hypercube_tree_test
   fastlib
   mlpack
   mlpack_contrib
)

add_executable(proximity_project
   main.cc
)
target_link_libraries(proximity_project
   fastlib
   mlpack
   mlpack_contrib
)

add_executable(kfold_splitter
   kfold_splitter.h
   kfold_splitter.cc
)
target_link_libraries(kfold_splitter
   fastlib
   mlpack
   mlpack_contrib
)
