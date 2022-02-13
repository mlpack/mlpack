# Define the files we need to compile
# Anything not in this list will not be compiled into mlpack.
set(SOURCES
  holt_winters.hpp
  holt_winters_impl.hpp
)

# Add directory name to sources.
set(DIR_SRCS)
foreach(file ${SOURCES})
  set(DIR_SRCS ${DIR_SRCS} ${CMAKE_CURRENT_SOURCE_DIR}/${file})
endforeach()
# Append sources (with directory name) to list of all mlpack sources (used at
# the parent scope).
set(MLPACK_SRCS ${MLPACK_SRCS} ${DIR_SRCS} PARENT_SCOPE)

add_cli_executable(holt_winters)
add_python_binding(holt_winters)
add_julia_binding(holt_winters)
add_go_binding(holt_winters)
add_r_binding(holt_winters)
add_markdown_docs(holt_winters "cli;python;julia;go;r" "")