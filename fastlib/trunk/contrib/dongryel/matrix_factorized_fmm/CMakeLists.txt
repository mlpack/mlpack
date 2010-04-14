cmake_minimum_required(VERSION 2.8)

# Define the files we need to compile
# Anything not in this list will not be compiled into the output library
# Do not include test programs here
set(SOURCES
   matrix_factorized_fmm.h
   matrix_factorized_fmm_impl.h
   matrix_factorized_fmm_stat.h
#   naive_kde.h
   matrix_factorized_local_expansion.h
   matrix_factorized_local_expansion_impl.h
   matrix_factorized_farfield_expansion.h
   matrix_factorized_farfield_expansion_impl.h
   cur_decomposition.h
   matrix_factorized_kernel_aux.h
)

# add directory name to sources
set(DIR_SRCS)
foreach(file ${SOURCES})
  set(DIR_SRCS ${DIR_SRCS} ${CMAKE_CURRENT_SOURCE_DIR}/${file})
endforeach()
# append sources (with directory name) to list of all MLPACK sources (used at the parent scope)
set(MLPACK_CONTRIB_SRCS ${MLPACK_CONTRIB_SRCS} ${DIR_SRCS} PARENT_SCOPE)

# executable
add_executable(matrix_factorized_fmm
   matrix_factorized_fmm_main.cc
)
# link dependencies of executable
target_link_libraries(matrix_factorized_fmm
   fastlib
   mlpack
   mlpack_contrib
)
