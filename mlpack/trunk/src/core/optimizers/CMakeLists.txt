cmake_minimum_required(VERSION 2.8)

set(DIRS
  aug_lagrangian
  lbfgs
)

foreach(dir ${DIRS})
  add_subdirectory(${dir})
endforeach()

set(MLPACK_SRCS ${MLPACK_SRCS} PARENT_SCOPE)
