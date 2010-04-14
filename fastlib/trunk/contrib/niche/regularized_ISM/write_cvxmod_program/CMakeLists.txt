cmake_minimum_required(VERSION 2.8)

# Define the files we need to compile
# Anything not in this list will not be compiled into the output library
# Do not include test programs here
set(SOURCES
   # no sources
)

# add directory name to sources
set(DIR_SRCS)
foreach(file ${SOURCES})
  set(DIR_SRCS ${DIR_SRCS} ${CMAKE_CURRENT_SOURCE_DIR}/${file})
endforeach()
# append sources (with directory name) to list of all MLPACK sources (used at the parent scope)
set(MLPACK_CONTRIB_SRCS ${MLPACK_CONTRIB_SRCS} ${DIR_SRCS} PARENT_SCOPE)

# executable
add_executable(write_cvxmod_prog
   write_cvxmod_prog.cc
)
# link dependencies of executable
target_link_libraries(write_cvxmod_prog
   fastlib
   mlpack
   mlpack_contrib
)

add_executable(extract_dists
   extract_dists.cc
)
target_link_libraries(extract_dists
   fastlib
   mlpack
   mlpack_contrib
)
