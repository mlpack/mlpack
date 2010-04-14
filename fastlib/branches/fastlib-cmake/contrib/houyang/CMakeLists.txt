cmake_minimum_required(VERSION 2.8)

# we just want to recurse into the child directories here
set(DIRS
   allknn_balltree
   allknn_kdtree
   regmin
   rvm
#   svm  ## requires mmanager
)

foreach(dir ${DIRS})
   add_subdirectory(${dir})
endforeach()

set(MLPACK_CONTRIB_SRCS ${MLPACK_CONTRIB_SRCS} PARENT_SCOPE)
