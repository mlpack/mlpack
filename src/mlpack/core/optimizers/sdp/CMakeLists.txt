set(SOURCES
  lrsdp.hpp
  lrsdp_impl.hpp
  lrsdp_function.hpp
  lrsdp_function_impl.hpp
  primal_dual.hpp
  primal_dual_impl.hpp
  sdp.hpp
  sdp_impl.hpp
)

set(DIR_SRCS)
foreach(file ${SOURCES})
  set(DIR_SRCS ${DIR_SRCS} ${CMAKE_CURRENT_SOURCE_DIR}/${file})
endforeach()

set(MLPACK_SRCS ${MLPACK_SRCS} ${DIR_SRCS} PARENT_SCOPE)
