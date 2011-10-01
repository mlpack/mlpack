cmake_minimum_required(VERSION 2.8)

# Define the files we need to compile
# Anything not in this list will not be compiled into the output library
# Do not include test programs here
set(SOURCES
   # approxnn
   approx_nn.h
   # approxnn_dual
   approx_nn_dual.h
   approx_nn_dual.cc
)

# add directory name to sources
set(DIR_SRCS)
foreach(file ${SOURCES})
  set(DIR_SRCS ${DIR_SRCS} ${CMAKE_CURRENT_SOURCE_DIR}/${file})
endforeach()
# append sources (with directory name) to list of all MLPACK sources (used at the parent scope)
set(MLPACK_CONTRIB_SRCS ${MLPACK_CONTRIB_SRCS} ${DIR_SRCS} PARENT_SCOPE)


# the first single tree trial
add_executable(approx_nn
   EXCLUDE_FROM_ALL
   main.cc
)
target_link_libraries(approx_nn
   fastlib
   mlpack
   mlpack_contrib
)

# complete main for RANN with kNN classification (I think)
add_executable(approx_nn_dual
   EXCLUDE_FROM_ALL
   main_dual.cc
)
target_link_libraries(approx_nn_dual
   fastlib
   mlpack
   mlpack_contrib
)

# compute the pairwise distances (and ranks) for given datasets
add_executable(pd_out
   EXCLUDE_FROM_ALL
   pairwise_dists.cc
)
target_link_libraries(pd_out
   fastlib
)

# cleaned up main for RANN
add_executable(rann_main
   EXCLUDE_FROM_ALL
   rann_main.cc
)
target_link_libraries(rann_main
   fastlib
   mlpack
   mlpack_contrib
)