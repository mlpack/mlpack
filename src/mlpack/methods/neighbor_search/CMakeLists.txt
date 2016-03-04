# Define the files we need to compile.
# Anything not in this list will not be compiled into mlpack.
set(SOURCES
  neighbor_search.hpp
  neighbor_search_impl.hpp
  neighbor_search_rules.hpp
  neighbor_search_rules_impl.hpp
  neighbor_search_stat.hpp
  ns_model.hpp
  ns_model_impl.hpp
  ns_traversal_info.hpp
  sort_policies/nearest_neighbor_sort.hpp
  sort_policies/nearest_neighbor_sort.cpp
  sort_policies/nearest_neighbor_sort_impl.hpp
  sort_policies/furthest_neighbor_sort.hpp
  sort_policies/furthest_neighbor_sort.cpp
  sort_policies/furthest_neighbor_sort_impl.hpp
  typedef.hpp
  unmap.hpp
  unmap.cpp
)

# Add directory name to sources.
set(DIR_SRCS)
foreach(file ${SOURCES})
  set(DIR_SRCS ${DIR_SRCS} ${CMAKE_CURRENT_SOURCE_DIR}/${file})
endforeach()
# Append sources (with directory name) to list of all mlpack sources (used at
# the parent scope).
set(MLPACK_SRCS ${MLPACK_SRCS} ${DIR_SRCS} PARENT_SCOPE)

add_executable(mlpack_knn
  allknn_main.cpp
)
target_link_libraries(mlpack_knn
  mlpack
)

add_executable(mlpack_kfn
  allkfn_main.cpp
)
target_link_libraries(mlpack_kfn
  mlpack
)

install(TARGETS mlpack_knn mlpack_kfn RUNTIME DESTINATION bin)
