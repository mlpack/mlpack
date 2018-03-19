set(SOURCES
  add_evaluate.hpp
  add_gradient.hpp
  add_evaluate_with_gradient.hpp
  add_decomposable_evaluate.hpp
  add_decomposable_gradient.hpp
  add_decomposable_evaluate_with_gradient.hpp
)

set(DIR_SRCS)
foreach(file ${SOURCES})
  set(DIR_SRCS ${DIR_SRCS} ${CMAKE_CURRENT_SOURCE_DIR}/${file})
endforeach()

set(MLPACK_SRCS ${MLPACK_SRCS} ${DIR_SRCS} PARENT_SCOPE)
