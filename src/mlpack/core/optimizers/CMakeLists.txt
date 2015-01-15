set(DIRS
  aug_lagrangian
  lbfgs
  lrsdp
  sa
  sdp
  sgd
)

foreach(dir ${DIRS})
  add_subdirectory(${dir})
endforeach()

set(MLPACK_SRCS ${MLPACK_SRCS} PARENT_SCOPE)
