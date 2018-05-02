# Define the files we need to compile
# Anything not in this list will not be compiled into mlpack.
set(SOURCES
  cross_entropy_error.hpp
  cross_entropy_error_impl.hpp
  kl_divergence.hpp
  kl_divergence_impl.hpp
  mean_squared_error.hpp
  mean_squared_error_impl.hpp
  sigmoid_cross_entropy_error.hpp
  sigmoid_cross_entropy_error_impl.hpp
)

# Add directory name to sources.
set(DIR_SRCS)
foreach(file ${SOURCES})
    set(DIR_SRCS ${DIR_SRCS} ${CMAKE_CURRENT_SOURCE_DIR}/${file})
endforeach()
# Append sources (with directory name) to list of all mlpack sources (used at
# the parent scope).
set(MLPACK_SRCS ${MLPACK_SRCS} ${DIR_SRCS} PARENT_SCOPE)
