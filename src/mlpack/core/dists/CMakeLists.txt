# Define the files we need to compile.
# Anything not in this list will not be compiled into mlpack.
set(SOURCES
  discrete_distribution.hpp
  discrete_distribution.cpp
  gaussian_distribution.hpp
  gaussian_distribution.cpp
  laplace_distribution.hpp
  laplace_distribution.cpp
  regression_distribution.hpp
  regression_distribution.cpp
  gamma_distribution.hpp
  gamma_distribution.cpp
)

# add directory name to sources
set(DIR_SRCS)
foreach(file ${SOURCES})
  set(DIR_SRCS ${DIR_SRCS} ${CMAKE_CURRENT_SOURCE_DIR}/${file})
endforeach()
# Append sources (with directory name) to list of all mlpack sources (used at
# the parent scope).
set(MLPACK_SRCS ${MLPACK_SRCS} ${DIR_SRCS} PARENT_SCOPE)
