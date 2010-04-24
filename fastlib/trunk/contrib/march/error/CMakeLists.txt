cmake_minimum_required(VERSION 2.8)

# Define the files we need to compile
# Anything not in this list will not be compiled into the output library
# Do not include test programs here
set(SOURCES
   # error_tester
   naive_kernel_sum.h
   hybrid_error_analysis.h
   hybrid_error.h
   hybrid_error_stat.h
)

# add directory name to sources
set(DIR_SRCS)
foreach(file ${SOURCES})
  set(DIR_SRCS ${DIR_SRCS} ${CMAKE_CURRENT_SOURCE_DIR}/${file})
endforeach()
# append sources (with directory name) to list of all MLPACK sources (used at the parent scope)
set(MLPACK_CONTRIB_SRCS ${MLPACK_CONTRIB_SRCS} ${DIR_SRCS} PARENT_SCOPE)

# executable
add_executable(error_tester
   EXCLUDE_FROM_ALL
   hybrid_error.cc
)
# link dependencies of executable
target_link_libraries(error_tester
   fastlib
   mlpack
   mlpack_contrib
)

#add_executable(naive_sum
#   naive_kernel_sum.h
#   naive_kernel_sum.cc
#)
#target_link_libraries(naive_sum
#   fastlib
#   mlpack
#   mlpack_contrib
#)
