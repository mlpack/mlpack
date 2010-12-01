cmake_minimum_required(VERSION 2.8)

# This is not compiled into mlpack_contrib by default, since it appears to be custom one-time-use
# code that is not useful anywhere other than this one particular test

# Define the files we need to compile
# Anything not in this list will not be compiled into the output library
# Do not include test programs here
set(SOURCES
   # multi_tree_fock
   multi_tree_fock.h
   multi_tree_fock.cc
   basis_shell_tree.h
   basis_shell_tree.cc
   shell_tree_impl.h
   shell_tree_impl.cc
   matrix_tree_impl.h
   matrix_tree.cc
   matrix_tree.h
#   matrix_tree_bounds.cc  # in build.py but does not exist
   eri_bounds.h
   eri_bounds.cc
)

# add directory name to sources
set(DIR_SRCS)
foreach(file ${SOURCES})
  set(DIR_SRCS ${DIR_SRCS} ${CMAKE_CURRENT_SOURCE_DIR}/${file})
endforeach()
# append sources (with directory name) to list of all MLPACK sources (used at the parent scope)
set(MLPACK_CONTRIB_SRCS ${MLPACK_CONTRIB_SRCS} ${DIR_SRCS} PARENT_SCOPE)

# executable
add_executable(shell_tree_test
   basis_shell_tree.h
   basis_shell_tree.cc
   shell_tree_impl.h
   shell_tree_impl.cc
   shell_tree_test.cc
)
# link dependencies of executable
target_link_libraries(shell_tree_test
   fastlib
   mlpack
   mlpack_contrib
)

add_executable(matrix_tree_test
   basis_shell_tree.h
   basis_shell_tree.cc
   shell_tree_impl.h
   shell_tree_impl.cc
   matrix_tree.h
   matrix_tree.cc
   matrix_tree_impl.h
   matrix_tree_impl.cc
   matrix_tree_test.cc
)
target_link_libraries(matrix_tree_test
   fastlib
   mlpack
   mlpack_contrib
)
