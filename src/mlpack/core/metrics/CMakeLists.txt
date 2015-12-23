# Define the files we need to compile.
# Anything not in this list will not be compiled into mlpack.
set(SOURCES
  ip_metric.hpp
  ip_metric_impl.hpp
  lmetric.hpp
  lmetric_impl.hpp
  mahalanobis_distance.hpp
  mahalanobis_distance_impl.hpp
)

# add directory name to sources
set(DIR_SRCS)
foreach(file ${SOURCES})
  set(DIR_SRCS ${DIR_SRCS} ${CMAKE_CURRENT_SOURCE_DIR}/${file})
endforeach()
# Append sources (with directory name) to list of all mlpack sources (used at
# the parent scope).
set(MLPACK_SRCS ${MLPACK_SRCS} ${DIR_SRCS} PARENT_SCOPE)
