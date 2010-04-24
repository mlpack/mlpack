cmake_minimum_required(VERSION 2.8)

# Define the files we need to compile
# Anything not in this list will not be compiled into the output library
# Do not include test programs here
set(SOURCES
   kernel_vector_mult.h
   solvelinsys.h
)

# add directory name to sources
set(DIR_SRCS)
foreach(file ${SOURCES})
  set(DIR_SRCS ${DIR_SRCS} ${CMAKE_CURRENT_SOURCE_DIR}/${file})
endforeach()
# append sources (with directory name) to list of all MLPACK sources (used at the parent scope)
set(MLPACK_CONTRIB_SRCS ${MLPACK_CONTRIB_SRCS} ${DIR_SRCS} PARENT_SCOPE)

# executable
add_executable(kernel_vector_mult
   EXCLUDE_FROM_ALL
   kernel_vector_mult_main.cc
)
# link dependencies of executable
target_link_libraries(kernel_vector_mult
   fastlib
)

add_executable(solvelinsys
   EXCLUDE_FROM_ALL
   solvelinsys_main.cc
)
target_link_libraries(solvelinsys
   fastlib
   mlpack
   mlpack_contrib
)
