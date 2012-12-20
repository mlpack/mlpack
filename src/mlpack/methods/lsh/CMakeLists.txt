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


# The code to compute the approximate neighbor
# for the given query and reference sets
# with p-stable LSH
add_executable(lsh
  lsh_main.cpp
)
target_link_libraries(lsh
  ${MLPACK_LIBRARY}
)

# The code to compute the approximate neighbor
# for the given query and reference sets 
# with p-stable LSH and analyze the time-accuracy
# tradeoff
add_executable(lsh_analysis
  lsh_analysis_main.cpp
)
target_link_libraries(lsh_analysis
  ${MLPACK_LIBRARY}
  contrib_pram
)


# Testing the LSHSearch class 
add_executable(lsh_test
  lsh_test.cpp
)
target_link_libraries(lsh_test
  ${MLPACK_LIBRARY}
)

