# Define the files we need to compile.
# Anything not in this list will not be compiled into MLPACK.
set(SOURCES
  allow_empty_clusters.hpp
  elkan_kmeans.hpp
  elkan_kmeans_impl.hpp
  hamerly_kmeans.hpp
  hamerly_kmeans_impl.hpp
  kmeans.hpp
  kmeans_impl.hpp
  max_variance_new_cluster.hpp
  max_variance_new_cluster_impl.hpp
  naive_kmeans.hpp
  naive_kmeans_impl.hpp
  pelleg_moore_kmeans.hpp
  pelleg_moore_kmeans_impl.hpp
  pelleg_moore_kmeans_rules.hpp
  pelleg_moore_kmeans_rules_impl.hpp
  pelleg_moore_kmeans_statistic.hpp
  random_partition.hpp
  refined_start.hpp
  refined_start_impl.hpp
)

# Add directory name to sources.
set(DIR_SRCS)
foreach(file ${SOURCES})
  set(DIR_SRCS ${DIR_SRCS} ${CMAKE_CURRENT_SOURCE_DIR}/${file})
endforeach()
# Append sources (with directory name) to list of all MLPACK sources (used at
# the parent scope).
set(MLPACK_SRCS ${MLPACK_SRCS} ${DIR_SRCS} PARENT_SCOPE)

# The main K-Means executable.
add_executable(kmeans
  kmeans_main.cpp
)
target_link_libraries(kmeans
  mlpack
)
install(TARGETS kmeans RUNTIME DESTINATION bin)
