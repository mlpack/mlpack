set(SOURCES
  booth_function.hpp
  booth_function.cpp
  colville_function.hpp
  colville_function.cpp
  matyas_function.hpp
  matyas_function.cpp
  mc_cormick_function.hpp
  mc_cormick_function.cpp
  rastrigin_function.hpp
  rastrigin_function.cpp
  sphere_function.hpp
  sphere_function.cpp
  styblinski_tang_function.hpp
  styblinski_tang_function.cpp
)

set(DIR_SRCS)
foreach(file ${SOURCES})
  set(DIR_SRCS ${DIR_SRCS} ${CMAKE_CURRENT_SOURCE_DIR}/${file})
endforeach()

set(MLPACK_SRCS ${MLPACK_SRCS} ${DIR_SRCS} PARENT_SCOPE)
