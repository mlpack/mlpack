cmake_minimum_required(VERSION 2.8)

# Define the files we need to compile
# Anything not in this list will not be compiled into the output library
# Do not include test programs here
set(SOURCES
   bounds_aux.h
   farfield_expansion.h
   farfield_expansion_impl.h
   inverse_pow_dist_kernel.h
   inverse_pow_dist_kernel_aux.h
   kernel_aux.h
   local_expansion.h
   local_expansion_impl.h
   mult_farfield_expansion.h
   mult_farfield_expansion_impl.h
   mult_local_expansion.h
   mult_local_expansion_impl.h
   mult_series_expansion_aux.h
   series_expansion_aux.h
   series_expansion_aux.cc
)

# add directory name to sources
set(DIR_SRCS)
foreach(file ${SOURCES})
  set(DIR_SRCS ${DIR_SRCS} ${CMAKE_CURRENT_SOURCE_DIR}/${file})
endforeach()
# append sources (with directory name) to list of all MLPACK sources (used at the parent scope)
set(MLPACK_SRCS ${MLPACK_SRCS} ${DIR_SRCS} PARENT_SCOPE)

# main executable
add_executable(series_expansion_test
   series_expansion_test.cc
)
# link dependencies of series_expansion
target_link_libraries(series_expansion_test
   mlpack
   fastlib
)
