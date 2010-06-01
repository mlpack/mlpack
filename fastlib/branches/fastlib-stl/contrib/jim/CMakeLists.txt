cmake_minimum_required(VERSION 2.8)

# we just want to recurse into the child directories here
set(DIRS
   cart
#   molecular_dynamics  ## does not compile: DHrectBound<2> has no member named 'width' (does not make sense, it does)
#   thor_md  ## same error as above
   thor_npoint
)

foreach(dir ${DIRS})
   add_subdirectory(${dir})
endforeach()

set(MLPACK_CONTRIB_SRCS ${MLPACK_CONTRIB_SRCS} PARENT_SCOPE)
