# Define the files we need to compile
# Anything not in this list will not be compiled into the output library
set(SOURCES
  data_dependent_random_initializer.hpp
  nothing_initializer.hpp
  random_initializer.hpp
  sparse_coding.hpp
  sparse_coding.cpp
  sparse_coding_impl.hpp
)

# add directory name to sources
set(DIR_SRCS)
foreach(file ${SOURCES})
  set(DIR_SRCS ${DIR_SRCS} ${CMAKE_CURRENT_SOURCE_DIR}/${file})
endforeach()

# append sources (with directory name) to list of all mlpack sources (used at the parent scope)
set(MLPACK_SRCS ${MLPACK_SRCS} ${DIR_SRCS} PARENT_SCOPE)

add_cli_executable(sparse_coding)
