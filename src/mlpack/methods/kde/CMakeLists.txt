# Define the files we need to compile.
# Anything not in this list will not be compiled into mlpack.
set(SOURCES
  kde.hpp
  kde_impl.hpp
  kde_rules.hpp
  kde_rules_impl.hpp
  kde_stat.hpp
  kde_model.hpp
  kde_model_impl.hpp
)

# Add directory name to sources.
set(DIR_SRCS)
foreach(file ${SOURCES})
  set(DIR_SRCS ${DIR_SRCS} ${CMAKE_CURRENT_SOURCE_DIR}/${file})
endforeach()
# Append sources (with directory name) to list of all mlpack sources (used at
# the parent scope).
set(MLPACK_SRCS ${MLPACK_SRCS} ${DIR_SRCS} PARENT_SCOPE)

add_cli_executable(kde)
add_python_binding(kde)
add_julia_binding(kde)
add_go_binding(kde)
add_r_binding(kde)
add_markdown_docs(kde "cli;python;julia;go;r" "misc. / other")
