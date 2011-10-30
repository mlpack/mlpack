cmake_minimum_required(VERSION 2.8)

# Define the files we need to compile
# Anything not in this list will not be compiled into the output library
# Do not include test programs here
set(SOURCES
   # Trees
   general_spacetree.h
   # Metric tree
   gen_metric_tree.h
   gen_metric_tree_impl.h
   gen_metric_tree_impl.cc

   cosine.h

   # Cosine tree
   dcosinebound.h
   gen_cosine_tree.h
   gen_cosine_tree_impl.h
   #gen_cosine_tree_impl.cc
   # Cone bound
   dconebound.h
   dconebound_impl.h
   # cone tree
   gen_cone_tree.h
   gen_cone_tree_impl.h
   gen_cone_tree_impl.cc
   # Max-inner-product search class
   exact_max_ip.h
   exact_max_ip.cc
   approx_max_ip.h
   approx_max_ip.cc
   # checking nn utils
   check_nn_utils.h
)

# add directory name to sources
set(DIR_SRCS)
foreach(file ${SOURCES})
  set(DIR_SRCS ${DIR_SRCS} ${CMAKE_CURRENT_SOURCE_DIR}/${file})
endforeach()
# append sources (with directory name) to list of all MLPACK sources (used at the parent scope)
set(MLPACK_CONTRIB_SRCS ${MLPACK_CONTRIB_SRCS} ${DIR_SRCS} PARENT_SCOPE)


# the single/dual tree trial
add_executable(max_ip_main
   EXCLUDE_FROM_ALL
   max_ip_main.cc
)
target_link_libraries(max_ip_main
   mlpack
   mlpack_contrib
)

# the single/dual tree approx trial
add_executable(approx_ip_main
   EXCLUDE_FROM_ALL
   approx_ip_main.cc
)
target_link_libraries(approx_ip_main
   mlpack
   mlpack_contrib
)

# the single/dual tree tester
add_executable(max_ip_tester
   EXCLUDE_FROM_ALL
   max_ip_tester.cc
)
target_link_libraries(max_ip_tester
   mlpack
   mlpack_contrib
)

# the single/dual tree approximate-search tester
add_executable(approx_tester
   EXCLUDE_FROM_ALL
   approx_tester.cc
)
target_link_libraries(approx_tester
   mlpack
   mlpack_contrib
)

# The cosine tree preliminary test file
add_executable(cosine_tree_test
   EXCLUDE_FROM_ALL
   cosine_tree_test.cc
)
target_link_libraries(cosine_tree_test
   mlpack
   mlpack_contrib
)

# The code to compute the rank for the given query 
# and reference sets 
add_executable(rank_computer
   EXCLUDE_FROM_ALL
   pairwise_dists.cc
)
target_link_libraries(rank_computer
   mlpack
)

# The code to compute the rank for the given query 
# and reference sets 
add_executable(rank_inverter
   EXCLUDE_FROM_ALL
   rank_matrix_inverter.cc
)
target_link_libraries(rank_inverter
   mlpack
   mlpack_contrib
)