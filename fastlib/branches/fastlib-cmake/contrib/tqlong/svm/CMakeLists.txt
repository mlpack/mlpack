cmake_minimum_required(VERSION 2.8)

# Define the files we need to compile
# Anything not in this list will not be compiled into the output library
# Do not include test programs here
set(SOURCES
   svm_projection.h
   project_simplex.cc
   min_quad_on_simplex.cc
   svm.h
   seqminopt.cc
   kernel_cache.cc
   index_set.cc
   kernel_function.cc
   ptswarmopt.cc
)

# add directory name to sources
set(DIR_SRCS)
foreach(file ${SOURCES})
  set(DIR_SRCS ${DIR_SRCS} ${CMAKE_CURRENT_SOURCE_DIR}/${file})
endforeach()
# append sources (with directory name) to list of all MLPACK sources (used at the parent scope)
set(MLPACK_CONTRIB_SRCS ${MLPACK_CONTRIB_SRCS} ${DIR_SRCS} PARENT_SCOPE)

# executable
add_executable(smo_tqlong_test
   smo_test.cc
)
# link dependencies of executable
target_link_libraries(smo_tqlong_test
   fastlib
   mlpack
   mlpack_contrib
)

#add_executable(pso_test
#   pso_test.cc
#)
#target_link_libraries(pso_test
#   fastlib
#   mlpack
#   mlpack_contrib
#)

add_executable(svm_projection_test
   svm_projection_test.cc
)
target_link_libraries(svm_projection_test
   fastlib
   mlpack
   mlpack_contrib
)
