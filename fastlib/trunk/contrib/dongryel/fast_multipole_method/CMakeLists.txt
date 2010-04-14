cmake_minimum_required(VERSION 2.8)

# Define the files we need to compile
# Anything not in this list will not be compiled into the output library
# Do not include test programs here
set(SOURCES
   # continuous_fmm
   continuous_fmm.h
   fmm_stat.h
   # fast_multipole_method
   fast_multipole_method.h
   fmm_stat.h
   # chebyshev_fit
   chebyshev_fit.h
   # multibody_force
   multibody_force_problem.h
   multibody_kernel.h
   # multibody_potential
   at_potential_kernel.h
   mbp_delta.h
   mbp_global.h
   mbp_kernel.h
   mbp_query_postponed.h
   mbp_query_result.h
   mbp_query_summary.h
   mbp_stat.h
   multibody_potential_problem.h
   three_body_gaussian_kernel.h
)

# add directory name to sources
set(DIR_SRCS)
foreach(file ${SOURCES})
  set(DIR_SRCS ${DIR_SRCS} ${CMAKE_CURRENT_SOURCE_DIR}/${file})
endforeach()
# append sources (with directory name) to list of all MLPACK sources (used at the parent scope)
set(MLPACK_CONTRIB_SRCS ${MLPACK_CONTRIB_SRCS} ${DIR_SRCS} PARENT_SCOPE)

# fast_multipole_method_bin executable
add_executable(fast_multipole_method
   fast_multipole_method_main.cc
)
target_link_libraries(fast_multipole_method
   fastlib
   mlpack
   mlpack_contrib
)

add_executable(continuous_fmm
   cfmm_main.cc
)
target_link_libraries(continuous_fmm
   fastlib
   mlpack
   mlpack_contrib
)

add_executable(multibody_force
   multibody_force_main.cc
)
target_link_libraries(multibody_force
   fastlib
   mlpack
   mlpack_contrib
)

add_executable(multibody_potential
   multibody_potential_main.cc
)
target_link_libraries(multibody_potential
   fastlib
   mlpack
   mlpack_contrib
)

# test executable
add_executable(chebyshev_fit_test
   chebyshev_fit_test.cc
)
target_link_libraries(chebyshev_fit_test
   fastlib
   mlpack
   mlpack_contrib
)
