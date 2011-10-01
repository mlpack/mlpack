cmake_minimum_required(VERSION 2.8)

# we just want to recurse into the child directories here
set(DIRS
   optimizers
)

foreach(dir ${DIRS})
   add_subdirectory(${dir})
endforeach()

set(MLPACK_CONTRIB_SRCS ${MLPACK_CONTRIB_SRCS} PARENT_SCOPE)
