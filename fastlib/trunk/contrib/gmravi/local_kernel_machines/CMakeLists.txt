cmake_minimum_required(VERSION 2.8)

# Define the files we need to compile
# Anything not in this list will not be compiled into the output library
# Do not include test programs here
set(SOURCES
   # svm
   opt_smo.h
#   opt_sgd.h  ## was in build.py but does not exist
   svm.h
   # svm_main.cc does not exist!
   # utils
   utils.h
   utils.cc
   # other things not properly documented or referenced in build.py
   ocas.h
   ocas.cc
   ocas_line_search.h
   ocas_line_search.cc
   ocas_smo.h
   ocas_smo.cc
   range_search.h
   my_crossvalidation.h
   local_kernel_machines_def.h
   local_kernel_machines_impl.h
)

# add directory name to sources
set(DIR_SRCS)
foreach(file ${SOURCES})
  set(DIR_SRCS ${DIR_SRCS} ${CMAKE_CURRENT_SOURCE_DIR}/${file})
endforeach()
# append sources (with directory name) to list of all MLPACK sources (used at the parent scope)
set(MLPACK_CONTRIB_SRCS ${MLPACK_CONTRIB_SRCS} ${DIR_SRCS} PARENT_SCOPE)

# executable
add_executable(local_kernel_machines
   local_kernel_machines_main.cc
)
# link dependencies of executable
target_link_libraries(local_kernel_machines
   fastlib
   mlpack
   mlpack_contrib
)

# test executables (not documented in build.py)
add_executable(ocas_smo_test
   ocas_smo_test.cc
)
target_link_libraries(ocas_smo_test
   fastlib
   mlpack
   mlpack_contrib
)

add_executable(range_search_test
   range_search_test.cc
)
target_link_libraries(range_search_test
   fastlib
   mlpack
   mlpack_contrib
)
