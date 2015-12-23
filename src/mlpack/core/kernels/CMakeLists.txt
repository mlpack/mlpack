# Define the files we need to compile.
# Anything not in this list will not be compiled into mlpack.
set(SOURCES
  cosine_distance.hpp
  cosine_distance_impl.hpp
  epanechnikov_kernel.hpp
  epanechnikov_kernel_impl.hpp
  epanechnikov_kernel.cpp
  example_kernel.hpp
  gaussian_kernel.hpp
  hyperbolic_tangent_kernel.hpp
  kernel_traits.hpp
  laplacian_kernel.hpp
  linear_kernel.hpp
  polynomial_kernel.hpp
  pspectrum_string_kernel.hpp
  pspectrum_string_kernel_impl.hpp
  pspectrum_string_kernel.cpp
  spherical_kernel.hpp
  triangular_kernel.hpp
)

# add directory name to sources
set(DIR_SRCS)
foreach(file ${SOURCES})
  set(DIR_SRCS ${DIR_SRCS} ${CMAKE_CURRENT_SOURCE_DIR}/${file})
endforeach()
# Append sources (with directory name) to list of all mlpack sources (used at
# the parent scope).
set(MLPACK_SRCS ${MLPACK_SRCS} ${DIR_SRCS} PARENT_SCOPE)
