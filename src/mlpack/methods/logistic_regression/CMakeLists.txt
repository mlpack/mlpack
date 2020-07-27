# Define the files we need to compile
# Anything not in this list will not be compiled into the output library
# Do not include test programs here
set(SOURCES
  logistic_regression.hpp
  logistic_regression_impl.hpp
  logistic_regression_function.hpp
  logistic_regression_function_impl.hpp
)

# add directory name to sources
set(DIR_SRCS)
foreach(file ${SOURCES})
  set(DIR_SRCS ${DIR_SRCS} ${CMAKE_CURRENT_SOURCE_DIR}/${file})
endforeach()
# append sources (with directory name) to list of all mlpack sources (used at
# the parent scope)
set(MLPACK_SRCS ${MLPACK_SRCS} ${DIR_SRCS} PARENT_SCOPE)

add_cli_executable(logistic_regression)
add_python_binding(logistic_regression)
add_julia_binding(logistic_regression)
add_go_binding(logistic_regression)
add_r_binding(logistic_regression)
add_markdown_docs(logistic_regression "cli;python;julia;go;r" "classification")
