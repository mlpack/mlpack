# Define the files we need to compile
# Anything not in this list will not be compiled into mlpack.
set(SOURCES
  kernel_pca.hpp
  kernel_pca_impl.hpp
)

add_subdirectory(kernel_rules)

# Add directory name to sources.
set(DIR_SRCS)
foreach(file ${SOURCES})
  set(DIR_SRCS ${DIR_SRCS} ${CMAKE_CURRENT_SOURCE_DIR}/${file})
endforeach()
# Append sources (with directory name) to list of all mlpack sources (used at
# the parent scope).
set(MLPACK_SRCS ${MLPACK_SRCS} ${DIR_SRCS} PARENT_SCOPE)

add_cli_executable(kernel_pca)
add_python_binding(kernel_pca)
add_julia_binding(kernel_pca)
add_go_binding(kernel_pca)
add_r_binding(kernel_pca)
add_markdown_docs(kernel_pca "cli;python;julia;go;r" "transformations")
