set(SOURCES
  cyclical_decay.hpp
  sgdr.hpp
  sgdr_impl.hpp
  snapshot_ensembles.hpp
  snapshot_sgdr.hpp
  snapshot_sgdr_impl.hpp
)

set(DIR_SRCS)
foreach(file ${SOURCES})
  set(DIR_SRCS ${DIR_SRCS} ${CMAKE_CURRENT_SOURCE_DIR}/${file})
endforeach()

set(MLPACK_SRCS ${MLPACK_SRCS} ${DIR_SRCS} PARENT_SCOPE)
