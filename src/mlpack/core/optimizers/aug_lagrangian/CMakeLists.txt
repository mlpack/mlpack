set(SOURCES
  aug_lagrangian.hpp
  aug_lagrangian_impl.hpp
  aug_lagrangian_function.hpp
  aug_lagrangian_function_impl.hpp
  aug_lagrangian_test_functions.hpp
  aug_lagrangian_test_functions.cpp
)

set(DIR_SRCS)
foreach(file ${SOURCES})
  set(DIR_SRCS ${DIR_SRCS} ${CMAKE_CURRENT_SOURCE_DIR}/${file})
endforeach()

set(MLPACK_SRCS ${MLPACK_SRCS} ${DIR_SRCS} PARENT_SCOPE)
