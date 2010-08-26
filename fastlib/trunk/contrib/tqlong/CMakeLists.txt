cmake_minimum_required(VERSION 2.8)

# we just want to recurse into the child directories here
set(DIRS
#   2dfactor  ## does not link; looking for nonexistent la:: functions
   gm
   graph
#   hmm  ## already exists elsewhere, appears to be a copy?
   kfs
#   mmf  ## a copy already exists in contrib/niche??
   #optim
   PassiveAggressive
#   quicsvd  ## does not link; looking for nonexistent la:: functions
   svm
#   affineNMF
   optimization
   GraphicalModel
)

foreach(dir ${DIRS})
   add_subdirectory(${dir})
endforeach()

set(MLPACK_CONTRIB_SRCS ${MLPACK_CONTRIB_SRCS} PARENT_SCOPE)
