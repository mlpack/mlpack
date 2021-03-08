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
  sort_policies/nearest_neighbor_sort.hpp
  sort_policies/nearest_neighbor_sort_impl.hpp
  sort_policies/furthest_neighbor_sort.hpp
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

# Add mlpack_knn and mlpack_kfn executables.
add_cli_executable(knn)
add_python_binding(knn)
add_julia_binding(knn)
add_go_binding(knn)
add_r_binding(knn)
add_markdown_docs(knn "cli;python;julia;go;r" "geometry")

add_cli_executable(kfn)
add_python_binding(kfn)
add_julia_binding(kfn)
add_go_binding(kfn)
add_r_binding(kfn)
add_markdown_docs(kfn "cli;python;julia;go;r" "geometry")
