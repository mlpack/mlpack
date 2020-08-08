# Define the files we need to compile
# Anything not in this list will not be compiled into the output library
# Do not include test programs here
set(SOURCES
  kernel_svm.hpp
  kernel_svm_impl.hpp
  kernel_svm_function.hpp
  kernel_svm_function_impl.hpp
)

# add directory name to sources
set(DIR_SRCS)
foreach(file ${SOURCES})
    set(DIR_SRCS ${DIR_SRCS} ${CMAKE_CURRENT_SOURCE_DIR}/${file})
endforeach()
# append sources (with directory name) to list of all mlpack sources (used at
# the parent scope)
set(MLPACK_SRCS ${MLPACK_SRCS} ${DIR_SRCS} PARENT_SCOPE)

add_cli_executable(kernel_svm)
add_python_binding(kernel_svm)
add_go_binding(kernel_svm)
add_julia_binding(kernel_svm)
add_markdown_docs(kernel_svm "cli;python;julia;go" "classification")
