# Define the files we need to compile
# Anything not in this list will not be compiled into mlpack.
set(SOURCES
  dbscan.hpp
  dbscan_impl.hpp
  random_point_selection.hpp
  ordered_point_selection.hpp
)

# Add directory name to sources.
set(DIR_SRCS)
foreach(file ${SOURCES})
  set(DIR_SRCS ${DIR_SRCS} ${CMAKE_CURRENT_SOURCE_DIR}/${file})
endforeach()
# Append sources (with directory name) to list of all mlpack sources (used at
# the parent scope).
set(MLPACK_SRCS ${MLPACK_SRCS} ${DIR_SRCS} PARENT_SCOPE)

add_cli_executable(dbscan)
add_python_binding(dbscan)
add_julia_binding(dbscan)
add_go_binding(dbscan)
add_r_binding(dbscan)
add_markdown_docs(dbscan "cli;python;julia;go;r" "clustering")
