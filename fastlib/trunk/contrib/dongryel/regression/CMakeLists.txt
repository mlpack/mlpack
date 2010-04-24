cmake_minimum_required(VERSION 2.8)

# Define the files we need to compile
# Anything not in this list will not be compiled into the output library
# Do not include test programs here
set(SOURCES
   # dense_lpr
   dense_lpr.h
   dense_lpr_impl.h
   epan_kernel_moment_info.h
   lpr_util.h
   matrix_util.h
   multi_index_util.h
   quick_prune_lpr.h
   relative_prune_lpr.h
   # krylov_lpr
     # epan_kernel_moment_info.h  # already included
   krylov_lpr.h
   krylov_lpr_setup_impl.h
   krylov_lpr_solver_impl.h
   krylov_lpr_test.h
     # lpr_util.h  # already included
   multi_conjugate_gradient.h
   multi_lanczos.h
   naive_lpr.h
)

# add directory name to sources
set(DIR_SRCS)
foreach(file ${SOURCES})
  set(DIR_SRCS ${DIR_SRCS} ${CMAKE_CURRENT_SOURCE_DIR}/${file})
endforeach()
# append sources (with directory name) to list of all MLPACK sources (used at the parent scope)
set(MLPACK_CONTRIB_SRCS ${MLPACK_CONTRIB_SRCS} ${DIR_SRCS} PARENT_SCOPE)

# executable
add_executable(krylov_lpr
   EXCLUDE_FROM_ALL
   krylov_lpr_main.cc
)
# link dependencies of executable
target_link_libraries(krylov_lpr
   fastlib
   mlpack
   mlpack_contrib
)

add_executable(dense_lpr
   EXCLUDE_FROM_ALL
   dense_lpr_main.cc
)
target_link_libraries(dense_lpr
   fastlib
   mlpack
   mlpack_contrib
)
