cmake_minimum_required(VERSION 2.8)

# Define the files we need to compile
# Anything not in this list will not be compiled into the output library
# Do not include test programs here
set(SOURCES
   # Tree files
   general_spacetree.h
   # Metric tree
   gen_metric_tree_impl.h
   gen_metric_tree.h
   # Cosine tree
   cosine.h
   dconebound_impl.h
   dconebound.h
   gen_cosine_tree_impl.h
   gen_cosine_tree.h
   # Exact maximum inner-product
   exact_max_ip.h
   exact_max_ip.cc
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
   #exact_max_ip.cc
   max_ip_main.cc
)
target_link_libraries(max_ip_main
   fastlib
   mlpack
   #mlpack_contrib
)

# The cosine tree preliminary test file
add_executable(cosine_tree_test
   EXCLUDE_FROM_ALL
   cosine_tree_test.cc
)
target_link_libraries(cosine_tree_test
   fastlib
   mlpack
)