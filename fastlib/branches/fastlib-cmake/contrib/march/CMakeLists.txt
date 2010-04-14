cmake_minimum_required(VERSION 2.8)

# we just want to recurse into the child directories here
set(DIRS
   emst_cover
   error
#   fock_matrix  ## awaiting resolution of #10 (requires libint)
#   hf
   npt
#   scf  ## awaiting resolution of #10 (requires libint)
)

foreach(dir ${DIRS})
   add_subdirectory(${dir})
endforeach()

set(MLPACK_CONTRIB_SRCS ${MLPACK_CONTRIB_SRCS} PARENT_SCOPE)
