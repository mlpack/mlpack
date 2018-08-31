add_library(mlpack_julia_util
  julia_util.h
  julia_util.cpp)

target_link_libraries(mlpack_julia_util mlpack)

add_library(mlpack_julia_pca
  julia_pca.h
  julia_pca.cpp
)

target_link_libraries(mlpack_julia_pca mlpack)
