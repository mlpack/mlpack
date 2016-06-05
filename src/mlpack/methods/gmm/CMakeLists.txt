# Define the files we need to compile.
# Anything not in this list will not be compiled into mlpack.
set(SOURCES
  gmm.hpp
  gmm.cpp
  gmm_impl.hpp
  em_fit.hpp
  em_fit_impl.hpp
  no_constraint.hpp
  positive_definite_constraint.hpp
  diagonal_constraint.hpp
  eigenvalue_ratio_constraint.hpp
)

# Add directory name to sources.
set(DIR_SRCS)
foreach(file ${SOURCES})
  set(DIR_SRCS ${DIR_SRCS} ${CMAKE_CURRENT_SOURCE_DIR}/${file})
endforeach()
# Append sources (with directory name) to list of all mlpack sources (used at
# the parent scope).
set(MLPACK_SRCS ${MLPACK_SRCS} ${DIR_SRCS} PARENT_SCOPE)

add_cli_executable(gmm_train)
add_cli_executable(gmm_generate)
add_cli_executable(gmm_probability)
