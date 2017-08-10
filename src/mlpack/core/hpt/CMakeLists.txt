set(SOURCES
  cv_function.hpp
  cv_function_impl.hpp
  deduce_hp_types.hpp
  fixed.hpp
  hpt.hpp
  hpt_impl.hpp
)

set(DIR_SRCS)
foreach(file ${SOURCES})
  set(DIR_SRCS ${DIR_SRCS} ${CMAKE_CURRENT_SOURCE_DIR}/${file})
endforeach()

set(MLPACK_SRCS ${MLPACK_SRCS} ${DIR_SRCS} PARENT_SCOPE)
