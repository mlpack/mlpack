# Define the files we need to compile
# Anything not in this list will not be compiled into mlpack.
set(SOURCES
  batch_svd_method.hpp
  bias_svd_method.hpp
  nmf_method.hpp
  randomized_svd_method.hpp
  regularized_svd_method.hpp
  svd_complete_method.hpp
  svd_incomplete_method.hpp
  svdplusplus_method.hpp
)

# Add directory name to sources.
set(DIR_SRCS)
foreach(file ${SOURCES})
  set(DIR_SRCS ${DIR_SRCS} ${CMAKE_CURRENT_SOURCE_DIR}/${file})
endforeach()
# Append sources (with directory name) to list of all mlpack sources (used at
# the parent scope).
set(MLPACK_SRCS ${MLPACK_SRCS} ${DIR_SRCS} PARENT_SCOPE)
