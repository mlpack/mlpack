# Define the files we need to compile.
# Anything not in this list will not be compiled into mlpack.
set(SOURCES
  perceptron.hpp
  perceptron_impl.hpp
)

add_subdirectory(initialization_methods)
add_subdirectory(learning_policies)

# Add directory name to sources.
set(DIR_SRCS)
foreach(file ${SOURCES})
  set(DIR_SRCS ${DIR_SRCS} ${CMAKE_CURRENT_SOURCE_DIR}/${file})
endforeach()
# Append sources (with directory name) to list of all mlpack sources (used at
# the parent scope).
set(MLPACK_SRCS ${MLPACK_SRCS} ${DIR_SRCS} PARENT_SCOPE)

add_cli_executable(perceptron)
add_python_binding(perceptron)
add_julia_binding(perceptron)
add_go_binding(perceptron)
add_r_binding(perceptron)
add_markdown_docs(perceptron "cli;python;julia;go;r" "classification")
