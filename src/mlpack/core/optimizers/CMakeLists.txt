set(DIRS
aug_lagrangian
lbfgs
minibatch_sgd
rmsprop
sa
sdp
parallel_sgd
scd
)

foreach(dir ${DIRS})
add_subdirectory(${dir})
endforeach()

set(MLPACK_SRCS ${MLPACK_SRCS} PARENT_SCOPE)
