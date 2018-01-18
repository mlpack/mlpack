set(SOURCES
  decay_policies/constant_step.hpp
  decay_policies/exponential_backoff.hpp
  parallel_sgd.hpp
  parallel_sgd_impl.hpp
  sparse_test_function.hpp
)

set(DIR_SRCS)
foreach(file ${SOURCES})
  set(DIR_SRCS ${DIR_SRCS} ${CMAKE_CURRENT_SOURCE_DIR}/${file})
endforeach()

set(MLPACK_SRCS ${MLPACK_SRCS} ${DIR_SRCS} PARENT_SCOPE)
