cmake_minimum_required(VERSION 2.8)

# Define the files we need to compile
# Anything not in this list will not be compiled into the output library
# Do not include test programs here
set(SOURCES
   ridge_regression.h
   ridge_regression_impl.h
   ridge_regression_util.h    
)

# add directory name to sources
set(DIR_SRCS)
foreach(file ${SOURCES})
  set(DIR_SRCS ${DIR_SRCS} ${CMAKE_CURRENT_SOURCE_DIR}/${file})
endforeach()
# append sources (with directory name) to list of all MLPACK sources (used at the parent scope)
set(MLPACK_SRCS ${MLPACK_SRCS} ${DIR_SRCS} PARENT_SCOPE)

# test executable
add_executable(ridge_regression_test
   ridge_regression_test.cc
)
# link dependencies of test executable
target_link_libraries(ridge_regression_test
   mlpack
   fastlib
)

# main executable
add_executable(ridge_regression
   ridge_main.cc
)
# link dependencies of ridge_regression
target_link_libraries(ridge_regression
   mlpack
   fastlib
)

# dataset preprocessing
add_executable(dataset_preprocess
   dataset_preprocess.cc
)
target_link_libraries(dataset_preprocess
   mlpack
   fastlib
)
