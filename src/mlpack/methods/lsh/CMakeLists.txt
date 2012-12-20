cmake_minimum_required(VERSION 2.8)

# Define the files we need to compile
# Anything not in this list will not be compiled into the output library
# Do not include test programs here
set(SOURCES
  
  # LSH-search class
  lsh_search.hpp
  lsh_search_impl.hpp
)

# add directory name to sources
set(DIR_SRCS)
foreach(file ${SOURCES})
  set(DIR_SRCS ${DIR_SRCS} ${CMAKE_CURRENT_SOURCE_DIR}/${file})
endforeach()
# append sources (with directory name) to list of all MLPACK sources (used at the parent scope)
set(MLPACK_CONTRIB_SRCS ${MLPACK_CONTRIB_SRCS} ${DIR_SRCS} PARENT_SCOPE)


# The code to compute the rank-approximate neighbor
# for the given query and reference sets
add_executable(allkdann
  lsh_main.cpp
)
target_link_libraries(allkdann
  ${MLPACK_LIBRARY}
  contrib_pram
)
