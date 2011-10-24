cmake_minimum_required(VERSION 2.8)

# Define the files we need to compile
# Anything not in this list will not be compiled into the output library
# Do not include test programs here
set(SOURCES
   # sampling
   sampling.h
   sampling.cc
   # objective2
   objective2.h
   objective2.cc
   # optimization
   optimization.h
   optimization.cc
)

# add directory name to sources
set(DIR_SRCS)
foreach(file ${SOURCES})
  set(DIR_SRCS ${DIR_SRCS} ${CMAKE_CURRENT_SOURCE_DIR}/${file})
endforeach()
# append sources (with directory name) to list of all MLPACK sources (used at the parent scope)
set(MLPACK_CONTRIB_SRCS ${MLPACK_CONTRIB_SRCS} ${DIR_SRCS} PARENT_SCOPE)

# executable
add_executable(ddcm_test
   EXCLUDE_FROM_ALL
   ddcm_test.cc
)
# link dependencies of executable
target_link_libraries(ddcm_test
   fastlib
   mlpack
   mlpack_contrib
)
