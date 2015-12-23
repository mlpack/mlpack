# Define the files we need to compile
# Anything not in this list will not be compiled into mlpack.
set(SOURCES
  nmf_als.hpp
  nmf_mult_dist.hpp
  nmf_mult_div.hpp
  svd_batch_learning.hpp
  svd_incomplete_incremental_learning.hpp
  svd_complete_incremental_learning.hpp
)

# Add directory name to sources.
set(DIR_SRCS)
foreach(file ${SOURCES})
  set(DIR_SRCS ${DIR_SRCS} ${CMAKE_CURRENT_SOURCE_DIR}/${file})
endforeach()
# Append sources (with directory name) to list of all mlpack sources (used at
# the parent scope).
set(MLPACK_SRCS ${MLPACK_SRCS} ${DIR_SRCS} PARENT_SCOPE)
