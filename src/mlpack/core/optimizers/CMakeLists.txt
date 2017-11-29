set(DIRS
  ada_delta
  comp_ada_grad
  ada_grad
  adam
  aug_lagrangian
  cne
  fw
  gradient_descent
  grid_search
  lbfgs
  line_search
  proximal
  rmsprop
  sa
  sdp
  sgd
  smorms3
  spalera_sgd
)

foreach(dir ${DIRS})
  add_subdirectory(${dir})
endforeach()

set(MLPACK_SRCS ${MLPACK_SRCS} PARENT_SCOPE)
