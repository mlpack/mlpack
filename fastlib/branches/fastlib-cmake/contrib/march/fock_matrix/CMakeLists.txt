cmake_minimum_required(VERSION 2.8)

# This project has subdirectories; we must recurse into them
add_subdirectory(cfmm)
add_subdirectory(chem_reader)
add_subdirectory(fock_impl)
add_subdirectory(link)
add_subdirectory(multi_tree)
add_subdirectory(naive)
add_subdirectory(prescreening)

# Define the files we need to compile
# Anything not in this list will not be compiled into the output library
# Do not include test programs here
set(SOURCES
   # add_subdirectory commands have already added all of the subdir sources

   # fock_matrix_comparison
   fock_matrix_comparison.h
   fock_matrix_comparison.cc
)

# add directory name to sources
set(DIR_SRCS)
foreach(file ${SOURCES})
  set(DIR_SRCS ${DIR_SRCS} ${CMAKE_CURRENT_SOURCE_DIR}/${file})
endforeach()
# append sources (with directory name) to list of all MLPACK sources (used at the parent scope)
set(MLPACK_CONTRIB_SRCS ${MLPACK_CONTRIB_SRCS} ${DIR_SRCS} PARENT_SCOPE)

# executable
add_executable(naive_fock_comparison
#   naive_fock_matrix.h
   naive_comparison.cc
)
# link dependencies of executable
target_link_libraries(naive_fock_comparison
   fastlib
   mlpack
   mlpack_contrib
)

add_executable(old_fock_matrix
   cfmm/cfmm_coulomb.h
   fock_matrix_main.cc
)
target_link_libraries(old_fock_matrix
   fastlib
   mlpack
   mlpack_contrib
)

add_executable(fock_matrix
   fock_matrix_main.cc
)
target_link_libraries(fock_matrix
   fastlib
   mlpack
   mlpack_contrib
)
