# Define the files we need to compile
# Anything not in this list will not be compiled into the output library
set(SOURCES
   radical.hpp
   radical.cpp
)

# add directory name to sources
set(DIR_SRCS)
foreach(file ${SOURCES})
  set(DIR_SRCS ${DIR_SRCS} ${CMAKE_CURRENT_SOURCE_DIR}/${file})
endforeach()
# append sources (with directory name) to list of all mlpack sources (used at the parent scope)
set(MLPACK_SRCS ${MLPACK_SRCS} ${DIR_SRCS} PARENT_SCOPE)

add_cli_executable(radical)
add_python_binding(radical)
add_julia_binding(radical)
add_go_binding(radical)
add_r_binding(radical)
add_markdown_docs(radical "cli;python;julia;go;r" "transformations")
