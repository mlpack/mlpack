cmake_minimum_required(VERSION 2.8)

# Define the files we need to compile
# Anything not in this list will not be compiled into the output library
# Do not include test programs here
set(SOURCES
   # build.py did not include these in any library, but I have chosen to
   linear_regression_model.h
   linear_regression_model_dev.h
   linear_regression_result.h
   linear_regression_result_dev.h
   qr_least_squares.h
   qr_least_squares_dev.h
   givens_rotate.h
   givens_rotate_dev.h
)

# add directory name to sources
set(DIR_SRCS)
foreach(file ${SOURCES})
  set(DIR_SRCS ${DIR_SRCS} ${CMAKE_CURRENT_SOURCE_DIR}/${file})
endforeach()
# append sources (with directory name) to list of all MLPACK sources (used at the parent scope)
set(MLPACK_CONTRIB_SRCS ${MLPACK_CONTRIB_SRCS} ${DIR_SRCS} PARENT_SCOPE)

# test executable
add_executable(vif_test
   EXCLUDE_FROM_ALL
   vif_test.cc
)
# link dependencies of test executable
target_link_libraries(vif_test
   fastlib
   mlpack
   mlpack_contrib
)
