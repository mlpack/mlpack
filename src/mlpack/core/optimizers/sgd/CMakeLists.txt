set(SOURCES
  decay_policies/no_decay.hpp
  update_policies/gradient_clipping.hpp
  update_policies/momentum_update.hpp
  update_policies/nesterov_momentum_update.hpp
  update_policies/vanilla_update.hpp
  sgd.hpp
  sgd_impl.hpp
)

set(DIR_SRCS)
foreach(file ${SOURCES})
  set(DIR_SRCS ${DIR_SRCS} ${CMAKE_CURRENT_SOURCE_DIR}/${file})
endforeach()

set(MLPACK_SRCS ${MLPACK_SRCS} ${DIR_SRCS} PARENT_SCOPE)
