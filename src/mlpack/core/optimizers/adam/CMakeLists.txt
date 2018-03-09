set(SOURCES
  adam.hpp
  adam_impl.hpp
  adam_update.hpp
  adamax_update.hpp
  amsgrad_update.hpp
  nadam_update.hpp
  nadamax_update.hpp
  optimisticadam_update.hpp
)

set(DIR_SRCS)
foreach(file ${SOURCES})
  set(DIR_SRCS ${DIR_SRCS} ${CMAKE_CURRENT_SOURCE_DIR}/${file})
endforeach()

set(MLPACK_SRCS ${MLPACK_SRCS} ${DIR_SRCS} PARENT_SCOPE)
