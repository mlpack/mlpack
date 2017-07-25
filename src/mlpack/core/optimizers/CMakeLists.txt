set(DIRS
  ada_delta
  ada_grad
  adam
  aug_lagrangian
  fw
  gradient_descent
  lbfgs
  minibatch_sgd
  rmsprop
  sa
  sdp
  sgd
  smorms3
)

foreach(dir ${DIRS})
  add_subdirectory(${dir})
endforeach()

set(MLPACK_SRCS ${MLPACK_SRCS} PARENT_SCOPE)
