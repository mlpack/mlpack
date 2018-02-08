set(SOURCES
  booth_function.hpp
  booth_function.cpp
  bukin_function.hpp
  bukin_function.cpp
  colville_function.hpp
  colville_function.cpp
  drop_wave_function.hpp
  drop_wave_function.cpp
  easom_function.hpp
  easom_function.cpp
  eggholder_function.hpp
  eggholder_function.cpp
  generalized_rosenbrock_function.hpp
  generalized_rosenbrock_function.cpp
  matyas_function.hpp
  matyas_function.cpp
  mc_cormick_function.hpp
  mc_cormick_function.cpp
  rastrigin_function.hpp
  rastrigin_function.cpp
  rosenbrock_function.hpp
  rosenbrock_wood_function.hpp
  rosenbrock_wood_function.cpp
  rosenbrock_function.cpp
  schwefel_function.hpp
  schwefel_function.cpp
  sgd_test_function.hpp
  sgd_test_function.cpp
  sphere_function.hpp
  sphere_function.cpp
  styblinski_tang_function.hpp
  styblinski_tang_function.cpp
  wood_function.hpp
  wood_function.cpp
)

set(DIR_SRCS)
foreach(file ${SOURCES})
  set(DIR_SRCS ${DIR_SRCS} ${CMAKE_CURRENT_SOURCE_DIR}/${file})
endforeach()

set(MLPACK_SRCS ${MLPACK_SRCS} ${DIR_SRCS} PARENT_SCOPE)
