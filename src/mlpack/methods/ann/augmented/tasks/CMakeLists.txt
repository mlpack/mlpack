# Define the files we need to compile
# Anything not in this list will not be compiled into mlpack.
set(SOURCES
  tasks.hpp
  add.hpp
  copy.hpp
  score.hpp
  sort.hpp
)

# Append sources (with directory name) to list of all mlpack sources (used at
# the parent scope).
set(MLPACK_SRCS ${MLPACK_SRCS} ${DIR_SRCS} PARENT_SCOPE)

# Add directory name to sources.
set(MLPACK_SRCS ${MLPACK_SRCS} PARENT_SCOPE)
