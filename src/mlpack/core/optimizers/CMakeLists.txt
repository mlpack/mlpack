set(DIRS
  ada_delta
  ada_grad
  adam
  aug_lagrangian
  cne
  fw
  gradient_descent
  grid_search
  katyusha
  lbfgs
  line_search
  proximal
  rmsprop
  sa
  sarah
  sdp
  sgd
  smorms3
  svrg
)

foreach(dir ${DIRS})
  add_subdirectory(${dir})
endforeach()

set(MLPACK_SRCS ${MLPACK_SRCS} PARENT_SCOPE)
