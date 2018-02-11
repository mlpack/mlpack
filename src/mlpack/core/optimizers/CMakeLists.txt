set(DIRS
  ada_delta
  ada_grad
  adam
  aug_lagrangian
  cmaes
  cne
  fw
  gradient_descent
  grid_search
  iqn
  lbfgs
  line_search
  problems
  proximal
  parallel_sgd
  rmsprop
  sa
  scd
  sdp
  sgd
  sgdr
  smorms3
  spalera_sgd
  pso
)

foreach(dir ${DIRS})
  add_subdirectory(${dir})
endforeach()

set(MLPACK_SRCS ${MLPACK_SRCS} PARENT_SCOPE)
