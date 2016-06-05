# Define the files we need to compile.
# Anything not in this list will not be compiled into mlpack.
set(SOURCES
  # LSH-search class
  lsh_search.hpp
  lsh_search_impl.hpp
)

# Add directory name to sources.
set(DIR_SRCS)
foreach(file ${SOURCES})
  set(DIR_SRCS ${DIR_SRCS} ${CMAKE_CURRENT_SOURCE_DIR}/${file})
endforeach()
# Append sources (with directory name) to list of all mlpack sources (used at
# the parent scope).
set(MLPACK_SRCS ${MLPACK_SRCS} ${DIR_SRCS} PARENT_SCOPE)

# The code to compute the approximate neighbor for the given query and reference
# sets with p-stable LSH.
add_cli_executable(lsh)
