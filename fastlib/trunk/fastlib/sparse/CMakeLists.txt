cmake_minimum_required(VERSION 2.8)

# Define the files we need to compile
# Anything not in this list will not be compiled into the output library
# Do not include test programs here
set(SOURCES
   sparse_matrix.h
   sparse_matrix_impl.h
)

# add directory name to sources
set(DIR_SRCS)
foreach(file ${SOURCES})
  set(DIR_SRCS ${DIR_SRCS} ${CMAKE_CURRENT_SOURCE_DIR}/${file})
endforeach()
# append sources (with directory name) to list of all FASTLIB sources (used at the parent scope)
set(FASTLIB_SRCS ${FASTLIB_SRCS} ${DIR_SRCS} PARENT_SCOPE)

#add_library(sparse ${SOURCES})
# link dependencies of sparse
#target_link_libraries(sparse
#   base
#   ${TRILINOS_LIBS}
#   lapack
#)

# test executable
add_executable(sparse_test
   sparse_matrix_test.cc
)
set_target_properties(sparse_test PROPERTIES
   COMPILE_FLAGS -fexceptions # required for trilinos
)
# link dependencies of test executable
target_link_libraries(sparse_test
#   sparse
   fastlib
   ${TRILINOS_LIBS}
)
