set(DIRS
  adadelta
  adam
  aug_lagrangian
  cg
  lbfgs
  minibatch_sgd
  parallel_sgd
  rmsprop
  sa
  sdp
  sgd
)

foreach(dir ${DIRS})
  add_subdirectory(${dir})
endforeach()

set(MLPACK_SRCS ${MLPACK_SRCS} PARENT_SCOPE)
