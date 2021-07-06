# Define the files we need to compile
# Anything not in this list will not be compiled into the output library
# Do not include test programs here
set(SOURCES
  linear_regression.hpp
  linear_regression.cpp
)

# add directory name to sources
set(DIR_SRCS)
foreach(file ${SOURCES})
  set(DIR_SRCS ${DIR_SRCS} ${CMAKE_CURRENT_SOURCE_DIR}/${file})
endforeach()
# append sources (with directory name) to list of all mlpack sources (used at
# the parent scope)
set(MLPACK_SRCS ${MLPACK_SRCS} ${DIR_SRCS} PARENT_SCOPE)

add_type(linear_regression "regression")
group_bindings(linear_regression "fit;predict")
add_python_binding(linear_regression)

add_cli_executable(linear_regression_fit)
add_julia_binding(linear_regression_fit)
add_go_binding(linear_regression_fit)
add_r_binding(linear_regression_fit)
add_markdown_docs(linear_regression_fit "cli;python;julia;go;r" "regression")

add_cli_executable(linear_regression_predict)
add_julia_binding(linear_regression_predict)
add_go_binding(linear_regression_predict)
add_r_binding(linear_regression_predict)
add_markdown_docs(linear_regression_predict "cli;python;julia;go;r" "regression")
