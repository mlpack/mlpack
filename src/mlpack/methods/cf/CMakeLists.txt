# Define the files we need to compile
# Anything not in this list will not be compiled into mlpack.
set(SOURCES
  cf.hpp
  cf_impl.hpp
  cf_model.hpp
  cf_model_impl.hpp
  svd_wrapper.hpp
  svd_wrapper_impl.hpp
)

add_subdirectory(decomposition_policies)
add_subdirectory(interpolation_policies)
add_subdirectory(neighbor_search_policies)
add_subdirectory(normalization)

# Add directory name to sources.
set(DIR_SRCS)
foreach(file ${SOURCES})
  set(DIR_SRCS ${DIR_SRCS} ${CMAKE_CURRENT_SOURCE_DIR}/${file})
endforeach()
# Append sources (with directory name) to list of all mlpack sources (used at
# the parent scope).
set(MLPACK_SRCS ${MLPACK_SRCS} ${DIR_SRCS} PARENT_SCOPE)

add_cli_executable(cf)
add_python_binding(cf)
add_julia_binding(cf)
add_go_binding(cf)
add_r_binding(cf)
add_markdown_docs(cf "cli;python;julia;go;r" "misc. / other")
