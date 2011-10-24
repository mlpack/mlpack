cmake_minimum_required(VERSION 2.8)

# Define the files we need to compile
# Anything not in this list will not be compiled into the output library
# Do not include test programs here
set(SOURCES
   # naive_kde_local_polynomial
   naive_kde_local_polynomial.h
   vector_kernel.h
   naive_local_likelihood.h
   # density_crossvalidation
   cross_validation.h
   naive_kde.h
)

# add directory name to sources
set(DIR_SRCS)
foreach(file ${SOURCES})
  set(DIR_SRCS ${DIR_SRCS} ${CMAKE_CURRENT_SOURCE_DIR}/${file})
endforeach()
# append sources (with directory name) to list of all MLPACK sources (used at the parent scope)
set(MLPACK_CONTRIB_SRCS ${MLPACK_CONTRIB_SRCS} ${DIR_SRCS} PARENT_SCOPE)

# executable
add_executable(naive_kde_lp
   EXCLUDE_FROM_ALL
   naive_kde_local_polynomial_main.cc
)
# link dependencies of executable
target_link_libraries(naive_kde_lp
   fastlib
   mlpack
   mlpack_contrib
)

# test executables (not documented in build.py)
add_executable(crossvalidation
   EXCLUDE_FROM_ALL
   crossvalidation_density_estimates_main.cc
)
target_link_libraries(crossvalidation
   fastlib
   mlpack
   mlpack_contrib
)
