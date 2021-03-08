# Define the files we need to compile.
# Anything not in this list will not be compiled into mlpack.
set(SOURCES
  lmnn.hpp
  lmnn_impl.hpp
  lmnn_function.hpp
  lmnn_function_impl.hpp
  constraints.hpp
  constraints_impl.hpp
)

# Add directory name to sources.
set(DIR_SRCS)
foreach(file ${SOURCES})
  set(DIR_SRCS ${DIR_SRCS} ${CMAKE_CURRENT_SOURCE_DIR}/${file})
endforeach()
# Append sources (with directory name) to list of all mlpack sources (used at
# the parent scope).
set(MLPACK_SRCS ${MLPACK_SRCS} ${DIR_SRCS} PARENT_SCOPE)

add_cli_executable(lmnn)
add_python_binding(lmnn)
add_julia_binding(lmnn)
add_go_binding(lmnn)
add_r_binding(lmnn)
add_markdown_docs(lmnn "cli;python;julia;go;r" "transformations")
