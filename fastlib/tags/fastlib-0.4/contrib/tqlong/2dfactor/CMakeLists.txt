cmake_minimum_required(VERSION 2.8)

# Define the files we need to compile
# Anything not in this list will not be compiled into the output library
# Do not include test programs here
set(SOURCES
   2dfactor.h
#   2dPCA.h  ## in build.py but does not exist
)

# add directory name to sources
set(DIR_SRCS)
foreach(file ${SOURCES})
  set(DIR_SRCS ${DIR_SRCS} ${CMAKE_CURRENT_SOURCE_DIR}/${file})
endforeach()
# append sources (with directory name) to list of all MLPACK sources (used at the parent scope)
set(MLPACK_CONTRIB_SRCS ${MLPACK_CONTRIB_SRCS} ${DIR_SRCS} PARENT_SCOPE)

# executable
add_executable(2dfactor_test
   EXCLUDE_FROM_ALL
   2dfactor_test.cc
)
# link dependencies of executable
target_link_libraries(2dfactor_test
   fastlib
   mlpack
   mlpack_contrib
)
