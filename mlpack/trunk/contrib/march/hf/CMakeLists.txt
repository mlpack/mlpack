cmake_minimum_required(VERSION 2.8)

# Define the files we need to compile
# Anything not in this list will not be compiled into the output library
# Do not include test programs here
set(SOURCES
   # dual_tree_integrals
   dual_tree_integrals.h
   dual_tree_integrals.cc
   square_tree.h
   # scf_solver
   scf_solver.h
   naive_fock_matrix.h
   scf_solver.cc
)

# add directory name to sources
set(DIR_SRCS)
foreach(file ${SOURCES})
  set(DIR_SRCS ${DIR_SRCS} ${CMAKE_CURRENT_SOURCE_DIR}/${file})
endforeach()
# append sources (with directory name) to list of all MLPACK sources (used at the parent scope)
set(MLPACK_CONTRIB_SRCS ${MLPACK_CONTRIB_SRCS} ${DIR_SRCS} PARENT_SCOPE)

# executable
add_executable(dual_tree_integrals
   EXCLUDE_FROM_ALL
   dual_tree_integrals.h
   naive_fock_matrix.h
   dual_tree_integrals_main.cc
)
# link dependencies of executable
target_link_libraries(dual_tree_integrals
   fastlib
   mlpack
   mlpack_contrib
)

add_executable(dualtree_fock_matrix_test
   EXCLUDE_FROM_ALL
   dual_tree_integrals.h
   naive_fock_matrix.h
   fock_matrix_test.cc
)
target_link_libraries(dualtree_fock_matrix_test
   fastlib
   mlpack
   mlpack_contrib
)

add_executable(hartree-fock
   EXCLUDE_FROM_ALL
   hf.cc
)
target_link_libraries(hartree-fock
   fastlib
   mlpack
   mlpack_contrib
)
