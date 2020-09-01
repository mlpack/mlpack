# Define the files we need to compile.
# Anything not in this list will not be compiled into mlpack.
set(SOURCES
  mean_shift.hpp
  mean_shift_impl.hpp
)

# Add directory name to sources.
set(DIR_SRCS)
foreach(file ${SOURCES})
  set(DIR_SRCS ${DIR_SRCS} ${CMAKE_CURRENT_SOURCE_DIR}/${file})
endforeach()
# Append sources (with directory name) to list of all mlpack sources (used at
# the parent scope).
set(MLPACK_SRCS ${MLPACK_SRCS} ${DIR_SRCS} PARENT_SCOPE)

add_cli_executable(mean_shift)
add_python_binding(mean_shift)
add_julia_binding(mean_shift)
add_go_binding(mean_shift)
add_r_binding(mean_shift)
add_markdown_docs(mean_shift "cli;python;julia;go;r" "clustering")
